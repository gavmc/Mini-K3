from torch import nn
import torch

from config import model_config


class MiniK3(nn.Module):
    def __init__(self, vocab_size, v_dim, k_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, v_dim)

    def forward(self):
        pass