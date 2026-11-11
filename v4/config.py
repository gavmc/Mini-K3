from pydantic import BaseModel
from pathlib import Path

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
    muon_lr: float = 0.01
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1

    adam_lr: float = 0.001
    adam_betas: tuple = (0.9, 0.95)
    adam_weight_decay: float = 0

    grad_clip: float = 1.0


class SchedulerConfig(BaseModel):
    base_lr: float  # needs to come from the optim config
    warmup: int = 1000
    cooldown: int = 1000
    total: int = 3000


class DataConfig(BaseModel):
    seed: int = 42
    buffer_size: int = 100_000
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)
    total_tokens: int = 100_000_000

    bytes_per_token: float = 3.0
    max_file_bytes: int = 128_000
    min_int_score: int = 2
    num_threads: int = 32
    out_dir: Path = Path("sample")
    s3_bucket: str = "softwareheritage"

    def budget(self, i: int) -> int:
        return int(self.total_tokens * self.split[i])