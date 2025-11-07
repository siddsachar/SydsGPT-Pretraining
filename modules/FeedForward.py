import torch.nn as nn
from modules.GELU import GELU

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], 4 * config["embedding_dim"]),
            GELU(),
            nn.Linear(4 * config["embedding_dim"], config["embedding_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)