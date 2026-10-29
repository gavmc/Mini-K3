import torch
import torch.nn as nn
from torch.optim import Muon, AdamW

from config import OptimConfig
from model import MiniK3

def newton_schulz(G: torch.Tensor, steps=5, eps=1e-7):

    @torch.jit.script
    def newton_schulz_iter(X, a: float, b: float, c: float):
        A = torch.matmul(X, X.transpose(-2, -1))
        B = b * A + c * torch.matmul(A, A)
        return a * X + torch.matmul(B, X)

    device = G.device
    dtype = G.dtype

    G = G.contiguous()
    X = G.to(torch.bfloat16)

    is_tall = X.size(-2) > X.size(-1)
    if is_tall:
        X = X.transpose(-2, -1)

    norm = torch.linalg.norm(X, dim=(-2, -1), keepdim=True)
    X = torch.div(X, norm + eps)

    a, b, c = (3.4445, -4.7750, 2.0315)

    for _ in range(steps):
        X = newton_schulz_iter(X, a, b, c)

    if is_tall:
        X = X.transpose(-2, -1)

    return X.to(G.dtype)

def muon_update(grad, momentum, beta=0.95, steps=5, nesterov=True):
    momentum.mul_(beta).add_(grad, alpha=1-beta)

    if nesterov:
        update = grad * (1 - beta) + momentum * beta
    else:
        update = momentum.clone()

    shape = update.shape
    reshaped = False

    if update.ndim > 2:
        reshaped = True
        update = update.view(update.size(0), -1)

    update = newton_schulz(update, steps=steps)

    scale_factor = max(1, update.size(-2) / update.size(-1))**0.5
    update.mul_(scale_factor)

    if reshaped:
        update = update.view(shape)

    return update

class CustomMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.95, weight_decay=0, steps=5, nesterov=True):

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, steps=steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            steps = group['steps']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                if p.ndim >= 2:
                    update = muon_update(grad, state['momentum_buffer'], beta=momentum, steps=steps, nesterov=nesterov)
                    p.add_(update, alpha=-lr)

                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)

                    if nesterov:
                        update = grad + momentum * buf
                    else:
                        update = buf

                    p.add_(update, alpha=-lr)

        return loss

# ^ need to update to a per-head muon if I end up using it later


def init_optim(model: MiniK3, cfg: OptimConfig):
    adam_params = set()
    muon_params = set()

    for mname, m in model.named_modules():
        for p in m.parameters(recurse=False):
            if isinstance(m, nn.Embedding):
                adam_params.add(p)
            elif isinstance(m, nn.Conv1d):
                adam_params.add(p)
            elif p.ndim < 2:
                adam_params.add(p)
            elif mname == "output":
                adam_params.add(p)
            else:
                muon_params.add(p)


    muon_optim = Muon(muon_params, lr=cfg.muon_lr, weight_decay=cfg.muon_weight_decay, momentum=cfg.muon_momentum)
    adam_optim = AdamW(adam_params, lr=cfg.adam_lr, betas=cfg.adam_betas, weight_decay=cfg.adam_weight_decay)

    return muon_optim, adam_optim