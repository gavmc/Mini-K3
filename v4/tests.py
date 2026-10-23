import torch
import gc

from model import MiniK3
from config import Config


import csv
import time
from pathlib import Path

import torch.nn.functional as F



device = "cuda"




cfgs = [
    # ~150M paramater models

    Config(
        layers=28,
        d_model=512,
        heads=8,
        d_ffn=2304,
    ),
    Config(
        layers=24,
        d_model=576,
        heads=9,
        d_ffn=2304,
    ),
    Config(
        layers=20,
        d_model=640,
        heads=10,
        d_ffn=2560,
    ),

    # ~300M parameter models

    Config(
        layers=40,
        d_model=640,
        heads=10,
        d_ffn=2560,
    ),
    Config(
        layers=28,
        d_model=768,
        heads=12,
        d_ffn=3072,
    ),
    Config(
        layers=20,
        d_model=896,
        heads=14,
        d_ffn=3584,
    ),

    # ~500M parameter models

    Config(
        layers=36,
        d_model=896,
        heads=14,
        d_ffn=3584,
    ),
    Config(
        layers=28,
        d_model=1024,
        heads=16,
        d_ffn=4096,
    ),
]



BATCH_SIZE = 1
SEQ_LEN = 2048
WARMUP_STEPS = 5
MEASURE_STEPS = 20

RESULTS_PATH = Path("benchmark_results.csv")
device = "cuda"


def benchmark(cfg, run_id):
    # Match the proposed head dimensions and MLA latent ratio.
    cfg.dk = 64
    cfg.dv = 64
    cfg.d_head = 64
    cfg.dc = cfg.d_model // 4

    row = {
        "run": run_id,
        "layers": cfg.layers,
        "d_model": cfg.d_model,
        "heads": cfg.heads,
        "d_ffn": cfg.d_ffn,
        "dc": cfg.dc,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "total_params": "",
        "trainable_params": "",
        "tokens_per_second": "",
        "step_ms": "",
        "peak_allocated_gib": "",
        "status": "",
    }

    model = optimizer = tokens = ids = targets = None

    try:
        torch.manual_seed(0)

        # Count before transferring so counts survive a GPU OOM.
        model = MiniK3(cfg)

        row["total_params"] = sum(
            p.numel() for p in model.parameters()
        )
        row["trainable_params"] = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        model = model.to(device).train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            fused=True,
        )

        # Shift by one token for next-token prediction.
        tokens = torch.randint(
            0,
            cfg.vocab_size,
            (BATCH_SIZE, SEQ_LEN + 1),
            device=device,
        )

        ids = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()

        def train_step():
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                result = model(ids, use_cache=False)

                # Supports logits or (logits, cache).
                logits = (
                    result[0]
                    if isinstance(result, (tuple, list))
                    else result
                )

                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    targets.reshape(-1),
                )

            loss.backward()
            optimizer.step()

        print(
            f"\nRun {run_id}: "
            f"L={cfg.layers}, d={cfg.d_model}, "
            f"parameters={row['total_params'] / 1e6:.2f}M",
            flush=True,
        )

        # Includes first-use kernel compilation and optimizer allocation.
        for _ in range(WARMUP_STEPS):
            train_step()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        for _ in range(MEASURE_STEPS):
            train_step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_tokens = (
            BATCH_SIZE * SEQ_LEN * MEASURE_STEPS
        )

        row["tokens_per_second"] = round(
            total_tokens / elapsed, 2
        )
        row["step_ms"] = round(
            1000 * elapsed / MEASURE_STEPS, 2
        )
        row["peak_allocated_gib"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
        row["status"] = "ok"

        print(
            f"  {row['tokens_per_second']:,.0f} tok/s | "
            f"{row['step_ms']:,.1f} ms/step | "
            f"{row['peak_allocated_gib']:.2f} GiB peak",
            flush=True,
        )

    except torch.OutOfMemoryError:
        row["status"] = "OOM"
        print("  Out of GPU memory; continuing.", flush=True)

    finally:
        # Release this configuration before constructing the next.
        optimizer = model = tokens = ids = targets = None

    return row


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark requires BF16 support.")

    print("GPU:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print(
        f"Training benchmark: B={BATCH_SIZE}, T={SEQ_LEN}, "
        f"BF16 autocast, AdamW, no torch.compile"
    )

    results = []

    with RESULTS_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = None

        for i, cfg in enumerate(cfgs, start=1):
            row = benchmark(cfg, i)

            # Collect after benchmark() returns so its local references
            # and any OOM traceback have been released.
            gc.collect()
            torch.cuda.empty_cache()

            if writer is None:
                writer = csv.DictWriter(
                    file, fieldnames=list(row)
                )
                writer.writeheader()

            writer.writerow(row)
            file.flush()
            results.append(row)

    print("\nResults")
    print(
        f"{'Run':>4} {'Width':>6} {'Layers':>6} "
        f"{'Params M':>10} {'Tok/s':>12} "
        f"{'Peak GiB':>10} {'Status':>8}"
    )

    for row in results:
        params = (
            f"{row['total_params'] / 1e6:.2f}"
            if row["total_params"] != ""
            else "-"
        )
        throughput = (
            f"{row['tokens_per_second']:,.0f}"
            if row["status"] == "ok"
            else "-"
        )
        memory = (
            f"{row['peak_allocated_gib']:.2f}"
            if row["status"] == "ok"
            else "-"
        )

        print(
            f"{row['run']:>4} "
            f"{row['d_model']:>6} "
            f"{row['layers']:>6} "
            f"{params:>10} "
            f"{throughput:>12} "
            f"{memory:>10} "
            f"{row['status']:>8}"
        )

    print(f"\nSaved to {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
