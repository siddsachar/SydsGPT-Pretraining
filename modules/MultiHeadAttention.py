import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, context_length, num_heads, qkv_bias = False):
        super().__init__()
        assert output_dim % num_heads == 0, "Output dimension must be divisible by number of heads"
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.weight_query = nn.Linear(input_dim, output_dim, qkv_bias)
        self.weight_key = nn.Linear(input_dim, output_dim, qkv_bias)
        self.weight_value = nn.Linear(input_dim, output_dim, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.triu(torch.ones(context_length, context_length), diagonal = 1))
        self.output_projection = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        queries = self.weight_query(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        keys = self.weight_key(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        values = self.weight_value(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        attention_scores = queries @ keys.transpose(2,3)
        attention_scores.masked_fill_(self.causal_mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim = -1)
        attention_weights = self.dropout(attention_weights)
        context_vectors = self.output_projection((attention_weights @ values).transpose(1,2).contiguous().view(batch_size, num_tokens, self.output_dim))
        return context_vectors