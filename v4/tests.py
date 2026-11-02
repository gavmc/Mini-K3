from model import MiniK3

from config import Config, OptimConfig, SchedulerConfig
from optim import init_optim
from scheduler import WSDScheduler

import torch

import wandb

model_cfg = Config()
optim_cfg = OptimConfig()

model = MiniK3(model_cfg).to("cuda")
model.train()


muon_optim, adam_optim = init_optim(model, optim_cfg)


muon_sched_cfg = SchedulerConfig(base_lr=optim_cfg.muon_lr)
adam_sched_cfg = SchedulerConfig(base_lr=optim_cfg.adam_lr)

muon_sched = WSDScheduler(muon_optim, muon_sched_cfg)
adam_sched = WSDScheduler(adam_optim, adam_sched_cfg)



tokens = torch.tensor([[1, 12, 34, 56, 12, 34, 56, 2]], device="cuda")
input_ids = tokens[:, :-1].contiguous()
targets = tokens[:, 1:].contiguous()


run = wandb.init(
    project="MiniK3",
    config={
        "epochs": 50,
    }
)

run.watch(model)

for _ in range(50):
    muon_optim.zero_grad(set_to_none=True)
    adam_optim.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids, targets=targets)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)

    muon_optim.step()
    adam_optim.step()
    muon_sched.step()
    adam_sched.step()

    print(f"Loss: {loss.item():.4f}, gradient norm: {grad_norm.item():.4f}")

    run.log({"Loss": loss.item()})

run.finish()

