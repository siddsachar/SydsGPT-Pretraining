import torch
import torch.nn as nn
from modules.TransformerBlock import TransformerBlock
from modules.LayerNorm import LayerNorm

class SydsGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.position_embedding = nn.Embedding(config["context_length"], config["embedding_dim"])
        self.drop_embedding = nn.Dropout(config["dropout"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["num_layers"])])
        self.final_layer_norm = LayerNorm(config["embedding_dim"])
        self.output_projection = nn.Linear(config["embedding_dim"], config["vocab_size"], bias = False)
    
    def forward(self, input):
        batch_size, seq_length = input.shape
        token_embeddings = self.token_embedding(input)
        position_embeddings = self.position_embedding(torch.arange(seq_length, device=input.device))
        x = token_embeddings + position_embeddings
        x = self.drop_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.output_projection(x)
        return logits