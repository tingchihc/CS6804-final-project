import torch.nn as nn
from config import device
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration

class T5Encoder(nn.Module):
    def __init__(self, model_name='t5-small', max_length=512):
        super(T5Encoder, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.max_length = max_length

    def forward(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        outputs = self.model(input_ids=inputs.input_ids)
        return outputs.last_hidden_state


class T5Decoder(nn.Module):
    def __init__(self, model_name='t5-small', max_length=512):
        super(T5Decoder, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_length = max_length

    def forward(self, input_embeddings, target_text=None):
        if target_text is not None:
            # Encode the target_text for teacher forcing
            target_encoding = self.tokenizer(
                target_text,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            decoder_input_ids = target_encoding['input_ids']
            outputs = self.model(
                inputs_embeds=input_embeddings.to(device),
                labels=decoder_input_ids.to(device),
                #decoder_input_ids=decoder_input_ids,
                return_dict=True,
            )
        else:
            # Use the model's generate() method for autoregressive decoding
            outputs = self.model.generate(
                inputs_embeds=input_embeddings,
                max_new_tokens=self.max_length,
            )

        return outputs
        # return outputs.logits
