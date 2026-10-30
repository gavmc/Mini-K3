from pydantic import BaseModel


class Config(BaseModel):
    layers: int = 24
    d_model: int = 576
    heads: int = 9
    d_ffn: int = 2304
    dv: int = 64
    dk: int = 64

    vocab_size: int = 32_000
    g_min: int = -5
    dc: int = 144
    d_head: int = d_model // heads



class OptimConfig(BaseModel):
    muon_lr: float = 0.001
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0

    adam_lr: float = 0.001
    adam_betas: tuple = (0.9, 0.95)
    adam_weight_decay: float = 0

    grad_clip: float = 1.0


class SchedulerConfig(BaseModel):
    base_lr: float = 0.005
    warmup: int = 1000
    cooldown: int = 1000
    total: int = 3000