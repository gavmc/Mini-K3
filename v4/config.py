from pydantic import BaseModel
from pathlib import Path

class ModelConfig(BaseModel):
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
    seed: int = 0
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


class TrainConfig(BaseModel):
    model: str = "D150_C"
    seed: int = 0
    tokens: int = 100_000_000
    seq_len: int = 2048
    seqs_per_update: int = 40
    micro_batch: int = 10
    warmup_updates: int = 100
    cooldown_frac: float = 0.2
    eval_every: int = 100
    val_seqs: int = 240
    data_dir: Path = Path("data/tokenized/sample-100M")
    split: dict[str, float] = {"fineweb-edu": 0.70, "cosmopedia-v2": 0.15, "stack-edu-Python": 0.15}
    optim: OptimConfig = OptimConfig()
    profile: bool = False
    log: bool = True

    @property
    def num_updates(self) -> int:
        return self.tokens // (self.seqs_per_update * self.seq_len)

    @property
    def num_micro(self) -> int:
        return self.seqs_per_update // self.micro_batch

    @property
    def cooldown_updates(self) -> int:
        return round(self.cooldown_frac * self.num_updates)