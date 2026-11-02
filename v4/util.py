from config import Config


def check_config(cfg: Config):
    assert cfg.layers % 4 == 0, "# of layers must be divisible by 4 (3 kda + 1 mla)"
    assert cfg.d_head == cfg.d_model / cfg.heads, "d_head must = d_model / # heads"

