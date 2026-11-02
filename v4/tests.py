from model import MiniK3

from config import Config, OptimConfig, SchedulerConfig
from optim import init_optim
from scheduler import WSDScheduler

import torch
import wandb

batch_size = 1
seq_len = 2048

model_cfg = Config()
optim_cfg = OptimConfig()

model = MiniK3(model_cfg).to("cuda")
model.train()


muon_optim, adam_optim = init_optim(model, optim_cfg)


muon_sched_cfg = SchedulerConfig(base_lr=optim_cfg.muon_lr, warmup=0, cooldown=0)
adam_sched_cfg = SchedulerConfig(base_lr=optim_cfg.adam_lr, warmup=0, cooldown=0)

muon_sched = WSDScheduler(muon_optim, muon_sched_cfg)
adam_sched = WSDScheduler(adam_optim, adam_sched_cfg)



tokens = torch.randint(
    model_cfg.vocab_size,
    (batch_size, seq_len+1),
    device="cuda"
)
input_ids = tokens[:, :-1].contiguous()
targets = tokens[:, 1:].contiguous()

for _ in range(5):
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


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print("running test")

torch.cuda.synchronize()
start.record()

for _ in range(25):
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


end.record()

torch.cuda.synchronize()

elapsed_seconds = start.elapsed_time(end) / 1000
tokens_processed = batch_size * seq_len * 25
print(f"tok/s - {tokens_processed / elapsed_seconds:.1f}")


