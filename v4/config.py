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


class ModelConfig(BaseModel):
    vocab_size: int = 16_000
    v_dim: int = 128
    k_dim: int = 64



dtypes = DataTypes()
model_config = ModelConfig()