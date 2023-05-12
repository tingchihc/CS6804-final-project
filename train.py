import json
import torch
import numpy as np
from torch.optim import Adam
from itertools import islice
from T5_encoder import T5Decoder
from GNN import CustomGNNLayer
from config import *
from model import CustomModel
from competition_text_only_KG_val_dgl import build_up_nx_KG

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield zip(*batch)

# Load a checkpoint
ckpt_num, custom_model = load_latest()
if not custom_model:
    
    # Initialize the T5 encoder and decoder
    t5_decoder = T5Decoder(max_length=MAX_ANS).to(device)

    # Initialize the GNN layer
    gnn_layer = CustomGNNLayer(MAX_QUE, DIM_HID, MAX_ANS).to(device)

    # Create the custom model with the T5 encoder, GNN layer, and T5 decoder
    custom_model = CustomModel(gnn_layer, t5_decoder).to(device)
ckpt_num += 1

# Set up the optimizer and loss function
optimizer = Adam(custom_model.parameters(), lr=1e-4)
#softmax layer ?????
loss_function = torch.nn.CrossEntropyLoss()

# Implement the training loop
with open(root % 'train') as jj:
    data_len = len(json.load(jj)['data'])
skip = (ckpt_num*ckpt_freq) % data_len
count = 0
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}/{num_epochs}")
    for target_texts, target_pages, graphs in batched(build_up_nx_KG(root % 'train', skip, progress=True), 1):
        skip = 0
        
        graphs = [g.to(device) for g in graphs]  # Move the graphs to GPU if available

        # Zero the gradients
        optimizer.zero_grad(True)

        # Calculate the loss
        losses = []
        for target_text, target_page, graph in zip(target_texts, target_pages, graphs):

            # Forward pass
            # logits, page = custom_model(graph, target_text)
            loss, page = custom_model(graph, target_text)

            # target_tokens = custom_model.t5_decoder.tokenizer.batch_encode_plus(
            #     target_text,
            #     return_tensors='pt',
            #     padding='max_length',
            #     truncation=True,
            #     max_length=custom_model.t5_decoder.max_length
            # )
            # target_ids = target_tokens['input_ids']
            # loss = loss_function(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss = loss + ETA * loss_function(page, torch.tensor(target_page).to(device))
            # loss = outputs
            losses.append(loss)

        # Calculate the mean loss for multiple target texts
        mean_loss = torch.stack(losses).mean()

        # Backward pass
        mean_loss.backward()
        optimizer.step()

        # Save checkpoints
        count += 1
        if not count % ckpt_freq:
            count = 0
            torch.save({'model': custom_model}, f"{checkpoint_dir}/{ckpt_num}")
            ckpt_num += 1

    print(f"Loss: {mean_loss.item()}")