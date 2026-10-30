from model import MiniK3

from config import Config, OptimConfig, SchedulerConfig
from optim import init_optim
from scheduler import WSDScheduler

from torch.optim import Muon, AdamW
import torch.nn as nn

model_cfg = Config()
optim_cfg = OptimConfig()
sched_cfg = SchedulerConfig()

model = MiniK3(model_cfg)


muon_optim, adam_optim = init_optim(model, optim_cfg)


muon_sched = WSDScheduler(muon_optim, sched_cfg)
adam_sched = WSDScheduler(adam_optim, sched_cfg)

