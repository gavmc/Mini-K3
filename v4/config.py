from pydantic import BaseModel, ConfigDict
import triton.language as tl
import torch


tl_map = {
    "fp16": tl.float16,
    "fp32": tl.float32,
    "bf16": tl.bfloat16,
}

torch_map = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

class DataTypes(BaseModel):
    q: str = "bf16"
    k: str = "bf16"
    v: str = "bf16"
    g: str = "fp32"
    beta: str = "fp32"
    state: str = "fp32"
    output: str = "bf16"
    compute: str = "fp32"



dtypes = DataTypes()


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

