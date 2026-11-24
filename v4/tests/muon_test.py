import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from config import ModelConfig, SchedulerConfig, TrainConfig
from model.model import MiniK3
from optimization.optim import init_optim
from optimization.scheduler import WSDScheduler
from tests.runs.sample_train import split_source, train_update


def benchmark(name, cfg, model_cfg, data, train_end, warmup, steps):

    torch.manual_seed(cfg.seed)
    model = MiniK3(model_cfg).cuda()

    muon, adam = init_optim(model, cfg.optim)

    if name == "stock":
        muon = torch.optim.Muon(muon.param_groups[0]["params"], lr=cfg.optim.muon_lr, momentum=cfg.optim.muon_momentum, weight_decay=cfg.optim.muon_weight_decay)

    model.compile()

    schedule = dict(warmup=cfg.warmup_updates, cooldown=cfg.cooldown_updates, total=cfg.num_updates)
    muon_sched = WSDScheduler(muon, SchedulerConfig(base_lr=cfg.optim.muon_lr, **schedule))
    adam_sched = WSDScheduler(adam, SchedulerConfig(base_lr=cfg.optim.adam_lr, **schedule))

    print(f"{name}: {warmup} warmup updates, then {steps} timed updates", flush=True)

    for update in range(warmup + steps):
        if update == warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()

        loss, _ = train_update(update, model, cfg, muon, adam, muon_sched, adam_sched, data, train_end)

    torch.cuda.synchronize()
    elapsed = perf_counter() - start

    if not torch.isfinite(loss).item():
        raise ValueError(f"{name} ended with a non-finite loss")
    
    rate = steps * cfg.seqs_per_update * cfg.seq_len / elapsed
    peak = torch.cuda.max_memory_allocated() / 2**30

    print(f"{name}: {elapsed / steps:.3f} s/update, {rate:,.0f} tok/s, {peak:.2f} GiB peak allocated", flush=True)
    
    return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare stock and batched Muon on complete pilot training updates.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batched-first", action="store_true")
    args = parser.parse_args()
    if min(args.warmup, args.steps) < 1:
        parser.error("--warmup and --steps must both be positive")

    cfg = TrainConfig(log=False, profile=False)
    configs = json.loads(Path(__file__).with_name("model_configs.json").read_text())
    model_cfg = ModelConfig.model_validate(configs[cfg.model])
    data = {name: np.memmap(cfg.data_dir / f"{name}.bin", dtype="<u2", mode="r") for name in cfg.split}
    train_end = {name: split_source(arr, cfg.val_seqs, cfg.seq_len)[0] for name, arr in data.items()}
    torch.set_float32_matmul_precision("high")
    print(f"{cfg.model}, {torch.cuda.get_device_name()}, {cfg.seqs_per_update} sequences/update, "
          f"{cfg.seq_len} tokens/sequence, micro-batch {cfg.micro_batch}", flush=True)

    rates = {}
    order = ("batched", "stock") if args.batched_first else ("stock", "batched")
    for name in order:
        rates[name] = benchmark(name, cfg, model_cfg, data, train_end, args.warmup, args.steps)
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Batched Muon throughput change: {(rates['batched'] / rates['stock'] - 1):+.2%}")
