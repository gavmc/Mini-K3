from model import MiniK3

from config import Config, OptimConfig, SchedulerConfig
from optim import init_optim
from scheduler import WSDScheduler

import torch

model_cfg = Config()
optim_cfg = OptimConfig()
sched_cfg = SchedulerConfig()

model = MiniK3(model_cfg).to("cuda")
model.train()


muon_optim, adam_optim = init_optim(model, optim_cfg)


muon_sched = WSDScheduler(muon_optim, sched_cfg)
adam_sched = WSDScheduler(adam_optim, sched_cfg)



tokens = torch.tensor([[1, 12, 34, 56, 12, 34, 56, 2]], device="cuda")
input_ids = tokens[:, :-1].contiguous()
targets = tokens[:, 1:].contiguous()

muon_optim.zero_grad(set_to_none=True)
adam_optim.zero_grad(set_to_none=True)

for _ in range(2):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids, targets=targets)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)

    muon_optim.step()
    adam_optim.step()
    muon_sched.step()
    adam_sched.step()

    print(f"Loss: {loss.item():.4f}, gradient norm: {grad_norm.item():.4f}")


model.eval()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out, cache = model(input_ids)

print(out.shape)
print(torch.argmax(out[0][-1]).item())
print()


token = torch.tensor([[torch.argmax(out[0][-1])]], device="cuda")

out, _ = model(token, cache=cache, use_cache=True)

print(out.shape)
print(torch.argmax(out[0][-1]).item())
print()

t = torch.cat((input_ids, token), dim=-1)
print(t.shape)

out, _ = model(torch.cat((input_ids, token), dim=-1))

print(out.shape)
print(torch.argmax(out[0][-1]).item())