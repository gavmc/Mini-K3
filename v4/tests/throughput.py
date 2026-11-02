from model import MiniK3
from config import Config, OptimConfig, SchedulerConfig
from optim import init_optim
from scheduler import WSDScheduler

import torch
import json
from pathlib import Path
import gc

configs = json.loads(Path(__file__).with_name("model_configs.json").read_text())

warmup = 5
seq_len = 2048
epochs = 50
batch_sizes = [1, 2, 4, 8]
runs = ["D150_B", "D300_B"]

optim_cfg = OptimConfig()
muon_sched_cfg = SchedulerConfig(base_lr=optim_cfg.muon_lr, warmup=0, cooldown=0)
adam_sched_cfg = SchedulerConfig(base_lr=optim_cfg.adam_lr, warmup=0, cooldown=0)

def benchmark(run, batch_size):
    model_cfg = Config.model_validate(configs[run])

    model = MiniK3(model_cfg).to("cuda")
    model.train()

    muon_optim, adam_optim = init_optim(model, optim_cfg)
    muon_sched = WSDScheduler(muon_optim, muon_sched_cfg)
    adam_sched = WSDScheduler(adam_optim, adam_sched_cfg)

    tokens = torch.randint(
        model_cfg.vocab_size,
        (batch_size, seq_len+1),
        device="cuda"
    )
    input_ids = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()

    for _ in range(warmup):
        muon_optim.zero_grad(set_to_none=True)
        adam_optim.zero_grad(set_to_none=True)
    
    
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids, targets=targets)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)
    
        muon_optim.step()
        adam_optim.step()
        muon_sched.step()
        adam_sched.step()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start.record()

    for _ in range(epochs):
        muon_optim.zero_grad(set_to_none=True)
        adam_optim.zero_grad(set_to_none=True)
    
    
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids, targets=targets)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)
    
        muon_optim.step()
        adam_optim.step()
        muon_sched.step()
        adam_sched.step()

    end.record()
    torch.cuda.synchronize()

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    print(f"Peak allocated: {peak_allocated:.2f} GiB")
    print(f"Peak reserved: {peak_reserved:.2f} GiB")

    elapsed_seconds = start.elapsed_time(end) / 1000
    tokens_processed = batch_size * seq_len * epochs

    return tokens_processed / elapsed_seconds, peak_allocated, peak_reserved


f = open(Path(__file__).with_name("throughput.md"), "w")

for run in runs:
    for batch_size in batch_sizes:
        try:
            tok_s, peak_allocated, peak_reserved = benchmark(run, batch_size)

            print(f"Model: {run}  Batch Size: {batch_size}  Tok/s: {tok_s:.1f}")
            print(f"Peak Memory Allocated: {peak_allocated:.2f}  Reserved: {peak_reserved:.2f}")
            f.write(f"Model: {run}  Batch Size: {batch_size}  Tok/s: {tok_s:.1f}\n")
            f.write(f"Peak Memory Allocated: {peak_allocated:.2f}  Reserved: {peak_reserved:.2f}\n")
            f.flush()
        except torch.cuda.OutOfMemoryError:
            print(f"OOM: model={run}, batch_size={batch_size}")
            f.write(f"Model: {run}  Batch Size: {batch_size}  Status: OOM\n")
            f.flush()

        gc.collect()
        torch.cuda.empty_cache()


f.close()


