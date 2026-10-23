from torch import nn
import torch

from config import config as cfg
from layers import Block


class MiniK3(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers // 4)])
        self.final_norm = nn.RMSNorm(cfg.d_model)

        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, input_ids, cache=None, use_cache=False):
        x = self.embedding(input_ids)

        if cache is None:
            cache = [None] * len(self.blocks)

        new_cache = []

        for i, block in enumerate(self.blocks):
            x, block_cache = block(
                x,
                cache=cache[i],
                use_cache=use_cache,
            )
            new_cache.append(block_cache)

        logits = self.output(self.final_norm(x))

        return logits, new_cache if use_cache else None