from torch import nn
import torch

from config import model_config


class MiniK3(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.v_dim)

    def forward(self):
        pass