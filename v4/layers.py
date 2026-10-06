import torch
from torch import nn


class KDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, state, scale):
        pass

    @staticmethod
    def backward(ctx, d_output, d_final_state):
        pass


class KDA(nn.Module):
    def forward(self, x, initial_state):
        pass
