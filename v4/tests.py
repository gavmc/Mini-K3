from model import MiniK3

from config import Config

from torch.optim import Muon, AdamW
import torch.nn as nn

cfg = Config()
model = MiniK3(cfg)


adam_params = set()
muon_params = set()

for mname, m in model.named_modules():
    for pname, p in m.named_parameters(recurse=False):
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


muon_optim = Muon(muon_params)
adam_optim = AdamW(adam_params)


        