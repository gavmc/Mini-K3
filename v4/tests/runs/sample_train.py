import torch
import json
from pathlib import Path
import numpy as np
import wandb

from model.model import MiniK3
from config import ModelConfig, SchedulerConfig, TrainConfig
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


def get_batch(step, train_cfg, data, train_end):
    split = train_cfg.split
    seq_len = train_cfg.seq_len
    rng = np.random.default_rng([train_cfg.seed, step])

    names = rng.choice(list(split), size=train_cfg.seqs_per_update, p=list(split.values()))
    rows = []

    for name in names:
        start = rng.integers(0, train_end[name] - (seq_len + 1))
        rows.append(torch.from_numpy(data[name][start:start + seq_len + 1].astype(np.int64)))

    return torch.stack(rows)


@torch.no_grad()
def evaluate(model, val_sets, micro_batch, split):
    model.eval()
    losses = {}

    for name, seqs in val_sets.items():
        total = 0.0

        for micro in seqs.split(micro_batch):
            x = micro.pin_memory().to("cuda", non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                total += model(x[:, :-1], x[:, 1:]).item() * len(micro)

        losses[f"val_loss/{name}"] = total / len(seqs)

    losses["val_loss/mix"] = sum(split[n] * losses[f"val_loss/{n}"] for n in split)

    model.train()
    return losses


def train(model, train_cfg, muon_optim, adam_optim, muon_sched, adam_sched, val_sets, data, train_end):
    model.train()

    for update in range(train_cfg.num_updates):
        
        if update % train_cfg.eval_every == 0:
            wandb.log(evaluate(model, val_sets, train_cfg.micro_batch, train_cfg.split), step=update)

        update_loss = 0.0
        batch = get_batch(update, train_cfg, data, train_end)

        muon_optim.zero_grad(set_to_none=True)
        adam_optim.zero_grad(set_to_none=True)

        for micro in batch.split(train_cfg.micro_batch):
            x = micro.pin_memory().to("cuda", non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x[:, :-1], x[:, 1:]) * (len(micro) / train_cfg.seqs_per_update)

            loss.backward()
            update_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.optim.grad_clip)
        muon_optim.step()
        adam_optim.step()
        muon_sched.step()
        adam_sched.step()


        wandb.log({
            "loss": update_loss, 
            "grad_norm": grad_norm.item(), 
            "muon_lr": muon_sched.get_last_lr()[0], 
            "adam_lr": adam_sched.get_last_lr()[0],
            "tokens_seen": (update + 1) * train_cfg.seqs_per_update * train_cfg.seq_len
        }, step=update)

    wandb.log(evaluate(model, val_sets, train_cfg.micro_batch, train_cfg.split), step=train_cfg.num_updates)





if __name__ == "__main__":
    configs = json.loads(Path(__file__).parent.with_name("model_configs.json").read_text())

    train_cfg = TrainConfig()
    model_cfg = ModelConfig.model_validate(configs[train_cfg.model])
    optim_cfg = train_cfg.optim
    muon_sched_cfg = SchedulerConfig(base_lr=optim_cfg.muon_lr, warmup=train_cfg.warmup_updates, cooldown=train_cfg.cooldown_updates, total=train_cfg.num_updates)
    adam_sched_cfg = SchedulerConfig(base_lr=optim_cfg.adam_lr, warmup=train_cfg.warmup_updates, cooldown=train_cfg.cooldown_updates, total=train_cfg.num_updates)

    data = {name: np.memmap(train_cfg.data_dir / f"{name}.bin", dtype="<u2", mode="r") for name in train_cfg.split}

    train_end = {}
    val_sets = {}
    
    for name in train_cfg.split:
        train_end[name], val_sets[name] = split_source(data[name], train_cfg.val_seqs, train_cfg.seq_len)

    print("creating model")

    model = MiniK3(model_cfg).to("cuda")
    muon_optim, adam_optim = init_optim(model, optim_cfg)
    model.compile()

    print("compiled model")

    muon_sched = WSDScheduler(muon_optim, muon_sched_cfg)
    adam_sched = WSDScheduler(adam_optim, adam_sched_cfg)


    wandb.init(
        project="Mini-k3",
        config={
            "model": train_cfg.model,
            "num_params": sum(p.numel() for p in model.parameters()),
            "model_cfg": model_cfg.model_dump(),

            "optim_cfg": optim_cfg.model_dump(),
            "muon_sched_cfg": muon_sched_cfg.model_dump(),
            "adam_sched_cfg": adam_sched_cfg.model_dump(),

            "train_cfg": train_cfg.model_dump(mode="json"),
            "precision": "bf16 autocast",
            "compiled": True,

            "tokenizer": "tokenizer/tokenizer-32K.json",
            "source_tokens": {name: len(arr) for name, arr in data.items()},
            "train_end": train_end,

            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        }
    )

    train(
        model=model, 
        train_cfg=train_cfg,
        data=data,
        train_end=train_end,
        muon_optim=muon_optim, 
        adam_optim=adam_optim, 
        muon_sched=muon_sched, 
        adam_sched=adam_sched,
        val_sets=val_sets,
    )

    budget = f"{train_cfg.tokens // 1_000_000}M" if train_cfg.tokens % 1_000_000 == 0 else str(train_cfg.tokens)
    torch.save(model.state_dict(), Path("tests/runs") / f"{train_cfg.model}_{budget}.pt")
    wandb.finish()
