from modules.MultiHeadAttention import MultiHeadAttention
from modules.LayerNorm import LayerNorm
from modules.FeedForward import FeedForward
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            input_dim = config["embedding_dim"],
            output_dim = config["embedding_dim"],
            dropout = config["dropout"],
            context_length = config["context_length"],
            num_heads = config["num_heads"],
            qkv_bias = config["qkv_bias"])
        self.layer_norm1 = LayerNorm(config["embedding_dim"])
        self.feed_forward = FeedForward(config)
        self.layer_norm2 = LayerNorm(config["embedding_dim"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut
        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x