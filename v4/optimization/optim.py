import torch
import torch.nn as nn
from torch.optim import AdamW

from config import OptimConfig
from model.model import MiniK3

def newton_schulz(G: torch.Tensor, steps=5, eps=1e-7):

    single_matrix = G.ndim == 2
    X = G.to(torch.bfloat16)
    if single_matrix:
        X = X.unsqueeze(0)

    is_tall = X.size(-2) > X.size(-1)
    if is_tall:
        X = X.transpose(-2, -1)

    norm = torch.linalg.norm(X, dim=(-2, -1), keepdim=True)
    X = X / norm.clamp_min(eps)

    a, b, c = (3.4445, -4.7750, 2.0315)

    for _ in range(steps):
        A = torch.bmm(X, X.transpose(-2, -1))
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a)

    if is_tall:
        X = X.transpose(-2, -1)

    if single_matrix:
        X = X.squeeze(0)
    return X.to(G.dtype)


class CustomMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.95, weight_decay=0, steps=5, nesterov=True):

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, steps=steps, nesterov=nesterov)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                if p.ndim != 2:
                    raise ValueError(f"CustomMuon requires 2-D parameters, got shape {tuple(p.shape)}")

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

            matrices = {}
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.grad)

                key = (p.shape, p.device, p.dtype)
                matrices.setdefault(key, []).append(p)

            for params in matrices.values():
                grads = [p.grad for p in params]
                buffers = [self.state[p]['momentum_buffer'] for p in params]
                torch._foreach_lerp_(buffers, grads, 1 - momentum)
                directions = torch._foreach_lerp(grads, buffers, momentum) if nesterov else buffers

                updates = newton_schulz(torch.stack(directions).bfloat16(), steps=steps)
                rows, cols = params[0].shape
                adjusted_lr = lr * max(1, rows / cols)**0.5

                if weight_decay != 0:
                    torch._foreach_mul_(params, 1 - lr * weight_decay)
                torch._foreach_add_(params, updates.to(params[0].dtype).unbind(), alpha=-adjusted_lr)

        return loss



def init_optim(model: MiniK3, cfg: OptimConfig):
    adam_params = []
    adam_params_no_decay = []
    muon_params = []

    for mname, m in model.named_modules():
        for p in m.parameters(recurse=False):
            if isinstance(m, nn.Embedding):
                adam_params.append(p)
            elif isinstance(m, nn.Conv1d):
                adam_params.append(p)
            elif p.ndim < 2:
                if hasattr(p, "_no_weight_decay") and p._no_weight_decay == True:
                    adam_params_no_decay.append(p)
                elif isinstance(m, nn.RMSNorm):
                    adam_params_no_decay.append(p)
                else:
                    adam_params.append(p)
            elif mname == "output":
                continue # <- shares weights with embedding layer
            else:
                muon_params.append(p)

    muon_optim = CustomMuon(muon_params, lr=cfg.muon_lr, weight_decay=cfg.muon_weight_decay, momentum=cfg.muon_momentum)
    adam_optim = AdamW([
        {"params": adam_params, "weight_decay": cfg.adam_weight_decay}, 
        {"params": adam_params_no_decay, "weight_decay": 0.0}],
        lr=cfg.adam_lr, betas=cfg.adam_betas)

    return muon_optim, adam_optim