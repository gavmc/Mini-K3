import torch
from torch import nn


class KDA(nn.Module):
    def __init__(self, dk, dv, nh):
        super().__init__()
