from torch import nn
import torch
import math
from fla.modules.fused_linear_cross_entropy import fused_linear_cross_entropy_loss

from config import Config
from layers import Block

class MiniK3(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers // 4)])
        self.final_norm = nn.RMSNorm(cfg.d_model)

        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.output.weight = self.embedding.weight

        self.init_weights(cfg)

    @torch.no_grad
    def init_weights(self, cfg, init_std=0.02):
        initialized = set()

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

        residual_std = init_std / math.sqrt(2 * cfg.layers)

        for block in self.blocks:
            for attention in block.attns:
                nn.init.normal_(attention.o_proj.weight, mean=0.0, std=residual_std)

            for ffn in block.ffns:
                nn.init.normal_(ffn.down.weight, mean=0.0, std=residual_std)

    def forward(self, input_ids, targets=None, cache=None, use_cache=False):
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

        x = self.final_norm(x)

        if targets is not None:
            loss = fused_linear_cross_entropy_loss(
                x=x.reshape(-1, x.shape[-1]),
                target=targets.reshape(-1),
                weight=self.output.weight,
                bias=self.output.bias,
                ignore_index=-100,
                reduction="mean",
            )

            return loss

        logits = self.output(x)

        return logits, new_cache if use_cache else None