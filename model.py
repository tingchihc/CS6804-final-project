import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, gnn_layer, t5_decoder):
        super(CustomModel, self).__init__()
        self.gnn_layer = gnn_layer
        self.t5_decoder = t5_decoder

    def forward(self, graph_input, target_text=None):
        # Pass the output embeddings from the T5 encoder to the GNN layer
        # Implement the GNN layer as needed
        #page_answer, 
        gnn_output, page = self.gnn_layer(graph_input)
        gnn_output = gnn_output.transpose(-1, -2)

        # Pass the output from the GNN layer to the T5 decoder
        if target_text:
            # return torch.cat([self.t5_decoder(gnn_output, t)
            #     for t in target_text]), page
            return torch.stack([self.t5_decoder(gnn_output, t).loss
                for t in target_text]).mean(), page
        else:
            # class GNNOutput:
            #     last_hidden_state=gnn_output
            #     def __getitem__(self, _):
            #         return self.last_hidden_state
            #     def __len__(self): return 1
            # outputs = self.t5_decoder.model.generate(inputs_embeds=gnn_output, max_new_tokens=self.t5_decoder.max_length)
            return self.t5_decoder.tokenizer.decode(
                self.t5_decoder(gnn_output)[0], skip_special_tokens=True), \
                    page.argmax().item()