import torch
import json
from pathlib import Path
import numpy as np
import wandb

from model.model import MiniK3
from config import Config, OptimConfig, SchedulerConfig
from optimization.optim import init_optim
from optimization.scheduler import WSDScheduler



def split_source(arr, val_seqs, seq_len):
    need = val_seqs * (seq_len + 1)

    eots = np.flatnonzero(arr[:len(arr) - need] == 3)
    cut = int(eots[-1]) + 1

    windows = []
    for i in range(val_seqs):
        start = cut + i * (seq_len + 1)
        windows.append(arr[start:start + seq_len + 1])

    val = torch.from_numpy(np.stack(windows).astype(np.int64))
    return cut, val


def get_batch(step, batch_size, seed=0):
    rng = np.random.default_rng([seed, step])

    names = rng.choice(list(split), size=batch_size, p=list(split.values()))
    rows = []

    for name in names:
        start = rng.integers(0, train_end[name] - (seq_len + 1))
        rows.append(torch.from_numpy(data[name][start:start + seq_len + 1].astype(np.int64)))

    return torch.stack(rows)


@torch.no_grad()
def evaluate(model, val_sets, micro_batch):
    model.eval()
    losses = {}

    for name, seqs in val_sets.items():
        total = 0.0

        for micro in seqs.split(micro_batch):
            x = micro.pin_memory().to("cuda", non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                total += model(x[:, :-1], x[:, 1:]).item()

        losses[f"val_loss/{name}"] = total / (len(seqs) // micro_batch)

    losses["val_loss/mix"] = sum(split[n] * losses[f"val_loss/{n}"] for n in split)

    model.train()
    return losses


def train(model, batch_data, muon_optim, adam_optim, muon_sched, adam_sched, val_sets, eval_every=100):
    model.train()

    num_updates, seqs_per_update, micro_batch, num_micro = batch_data

    for update in range(num_updates):
        
        if update % eval_every == 0:
            wandb.log(evaluate(model, val_sets, micro_batch), step=update)

        update_loss = 0.0
        batch = get_batch(update, seqs_per_update)

        muon_optim.zero_grad(set_to_none=True)
        adam_optim.zero_grad(set_to_none=True)

        for micro in batch.split(micro_batch):
            x = micro.pin_memory().to("cuda", non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x[:, :-1], x[:, 1:]) / num_micro

            loss.backward()
            update_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)
        muon_optim.step()
        adam_optim.step()
        muon_sched.step()
        adam_sched.step()


        wandb.log({
            "loss": update_loss, 
            "grad_norm": grad_norm.item(), 
            "muon_lr": muon_sched.get_last_lr()[0], 
            "adam_lr": adam_sched.get_last_lr()[0],
           "tokens_seen": (update + 1) * seqs_per_update * seq_len
        }, step=update)

    wandb.log(evaluate(model, val_sets, micro_batch), step=num_updates)





if __name__ == "__main__":
    configs = json.loads(Path(__file__).parent.with_name("model_configs.json").read_text())

    model_version = "D150_C"
    cfg = Config.model_validate(configs[model_version])
    optim_cfg = OptimConfig()
    muon_sched_cfg = SchedulerConfig(base_lr=optim_cfg.muon_lr, warmup=round(0.005*1220), cooldown=round(0.2*1220), total=1220)
    adam_sched_cfg = SchedulerConfig(base_lr=optim_cfg.adam_lr, warmup=round(0.005*1220), cooldown=round(0.2*1220), total=1220)

    tokens = 100_000_000
    seq_len = 2048
    val_seqs = 240

    batch_size = 10
    seqs_per_update = 40
    micro_batch = 10
    num_micro = seqs_per_update // micro_batch
    num_updates = tokens // (seqs_per_update * seq_len)

    epochs = tokens / seq_len / batch_size
    split = {"fineweb-edu": 0.70, "cosmopedia-v2": 0.15, "stack-edu-Python": 0.15}
    data = {name: np.memmap(f"data/tokenized/sample-100M/{name}.bin", dtype="<u2", mode="r") for name in split}

    train_end = {}
    val_sets = {}
    
    for name in split:
        train_end[name], val_sets[name] = split_source(data[name], val_seqs, seq_len)

    print("creating model")

    model = MiniK3(cfg).to("cuda")
    muon_optim, adam_optim = init_optim(model, optim_cfg)
    model.compile()

    print("compiled model")

    muon_sched = WSDScheduler(muon_optim, muon_sched_cfg)
    adam_sched = WSDScheduler(adam_optim, adam_sched_cfg)


    wandb.init(
        project="Mini-k3",
        config={
            "model": model_version,
            "num_params": sum(p.numel() for p in model.parameters()),
            "model_cfg": cfg.model_dump(),

            "optim_cfg": optim_cfg.model_dump(),
            "muon_sched_cfg": muon_sched_cfg.model_dump(),
            "adam_sched_cfg": adam_sched_cfg.model_dump(),

            "tokens": tokens,
            "seq_len": seq_len,
            "seqs_per_update": seqs_per_update,
            "micro_batch": micro_batch,
            "num_micro": num_micro,
            "tokens_per_update": seqs_per_update * seq_len,
            "num_updates": num_updates,
            "precision": "bf16 autocast",
            "compiled": True,

            "data_dir": "data/tokenized/sample-100M",
            "tokenizer": "tokenizer/tokenizer-32K.json",
            "mix": split,
            "data_seed": 0,
            "source_tokens": {name: len(arr) for name, arr in data.items()},
            "train_end": train_end,
            "val_seqs": val_seqs,

            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        }
    )

    train(
        model=model, 
        batch_data=(num_updates, seqs_per_update, micro_batch, num_micro),
        muon_optim=muon_optim, 
        adam_optim=adam_optim, 
        muon_sched=muon_sched, 
        adam_sched=adam_sched,
        val_sets=val_sets,
    )

    torch.save(model.state_dict(), "tests/runs/D150_C_100M.pt")
    wandb.finish()
