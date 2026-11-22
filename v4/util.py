from config import ModelConfig


def check_config(cfg: ModelConfig):
    assert cfg.layers % 4 == 0, "# of layers must be divisible by 4 (3 kda + 1 mla)"
    assert cfg.d_head == cfg.d_model / cfg.heads, "d_head must = d_model / # heads"

