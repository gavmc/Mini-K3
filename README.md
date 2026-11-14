# Mini-K3

**Small language models built and trained from scratch on a single RTX 3090.**

Mini-K3 is an independent PyTorch research project exploring Kimi-inspired architectures on consumer hardware. V4 combines Kimi Delta Attention (KDA), gated multi-head latent attention (MLA), and dense SwiGLU FFNs. MoE and attention residuals are later experiments.

The practical question is which ideas improve a useful small model under a limited training budget. The target behaviors are instruction following, rewriting, short summarization, copying names and numbers, following list constraints, and clean stopping.

> **Status (2026-11-13):** The V4 model, Muon + AdamW optimizers, and WSD scheduler are implemented. Eight model configurations have been benchmarked, with three shortlisted. The pretraining sample has been downloaded and tokenized: 126.53M tokens stored as uint16, verified against every source document. The training loop, validation split, checkpoint/resume, evaluation, generation driver, and SFT remain to be built. No V4 language-quality result is available yet.

This is an educational research project and an independent implementation, with no affiliation with Moonshot AI.

## Goals and scope

- Train a useful small language model on one RTX 3090 with 24 GB VRAM.
- Use a context length of 2,048 tokens.
- Target approximately 20B pretraining tokens for the selected model; smaller pilots are additional unless explicitly budgeted otherwise.
- Compare width and depth using measured throughput, memory, and learning results.
- Establish a dense FFN baseline before experimenting with MoE or AttnRes.
- Use FLA for training and keep custom Triton kernels as educational references.

No final model configuration or training recipe has been selected.

## Versions

| Version | Purpose | Contents |
| --- | --- | --- |
| [`v1/`](v1/) | Early MoE experiments | Expert training and merging code |
| [`v2/`](v2/) | Earlier end-to-end iteration | Model, data preparation, pretraining, SFT, sampling, optimizers, and design notes |
| [`v3/`](v3/) | FLA-based architecture prototype | Hybrid KDA/MLA model, latent MoE, AttnRes, configuration, and synthetic benchmark |
| [`v4/`](v4/) | Current development | Dense KDA/MLA model, optimizers, scheduler, throughput benchmarks, tokenizer, and sample preparation |

These versions are separate experiments. Their configurations, checkpoints, and entry points are not interchangeable.

## V4 architecture

The model lives in [`v4/model/model.py`](v4/model/model.py), with layers in [`v4/model/layers.py`](v4/model/layers.py).

Each block contains three KDA mixers followed by one MLA mixer. Every mixer has a dense SwiGLU FFN after it. Each mixer and FFN has its own pre-RMSNorm and additive residual. `cfg.layers` counts mixer+FFN pairs, so the model builds `layers // 4` blocks. A final RMSNorm precedes the output, and the 32,000-token embedding and output projection share weights.

### Kimi Delta Attention

- Separate Q/K/V projections followed by short causal convolutions with kernel size 4.
- Per-head update strengths and per-key-channel decay gates computed in FP32.
- Query/key L2 normalization inside the FLA kernel.
- Per-head output RMSNorm and a learned sigmoid output gate.
- Explicit recurrent-state and convolution-state caches.

The training and prefill path uses [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) `chunk_kda`. Single-token evaluation with gradients disabled uses `fused_recurrent_kda`.

### Gated latent attention

MLA compresses each token to a shared normalized latent, reconstructs per-head keys and values, and applies causal scaled-dot-product attention followed by a sigmoid output gate. It uses no explicit positional encoding; positional information reaches it through the preceding KDA layers.

The cache stores latents. Keys and values for the full history are reconstructed on each call. Multi-token cache extensions use an offset causal mask.

### Model interface

```python
loss = model(input_ids, targets=targets)
logits, cache = model(input_ids, cache=cache, use_cache=True)
```

With targets, the model returns a scalar fused linear cross-entropy loss with `ignore_index=-100`. Callers shift targets once: `input_ids = tokens[:, :-1]`, `targets = tokens[:, 1:]`.

Caches are passed explicitly and are not stored on the model. FLA convolution caches may be mutated in place, so clone a cache before branching from a shared prefix. Causal behavior and full-sequence versus cached-decoding agreement still need recorded correctness checks.

## Shortlisted models and measured throughput

All eight candidate configurations remain in [`v4/tests/model_configs.json`](v4/tests/model_configs.json). The current shortlist is **D150_B, D150_C, and D300_C**.

Gavin's compiled desktop runs used synthetic tokens at context 2,048, BF16 autocast, optimizer steps, and one process per model/batch combination. The table shows the fastest tested batch for each shortlisted model, from [`throughput.md`](v4/tests/throughput/throughput.md).

| Candidate | Width | Mixer+FFN pairs | Heads | Parameters | Batch | Tokens/sec | Peak allocated GiB | Ideal 20B-token days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D150_B | 576 | 24 | 9 | 152,893,890 | 10 | 25,364.6 | 20.92 | 9.1 |
| D150_C | 640 | 20 | 10 | 158,661,750 | 10 | 26,829.3 | 19.78 | 8.6 |
| D300_C | 896 | 20 | 14 | 298,734,066 | 8 | 17,258.9 | 22.59 | 13.4 |

Both D150 models hit OOM at batch 12. D300_C at batch 8 leaves little headroom for a real trainer. Ideal days exclude data loading, evaluation, checkpointing, and downtime. These measurements establish speed and memory use; model quality has not been compared yet.

## Optimizers and scheduler

[`init_optim`](v4/optimization/optim.py) returns PyTorch Muon and AdamW. Muon receives most matrices; AdamW receives the tied embedding, convolution weights, and vectors. RMSNorm gains and tagged decay parameters have zero weight decay.

Current provisional defaults are Muon LR 0.01, momentum 0.95, weight decay 0.1; AdamW LR 0.001, betas (0.9, 0.95), weight decay 0; global gradient clipping at 1.0. These settings have not been tuned in learning pilots.

[`WSDScheduler`](v4/optimization/scheduler.py) implements linear warmup, a constant phase, and `1 - sqrt(progress)` cooldown to zero. Phase lengths count optimizer updates. Each optimizer needs its own scheduler with the corresponding `base_lr`. The benchmark uses no LR warmup; it does not establish a long-run recipe. Optimizer and scheduler resume has not been exercised.

## Tokenizer and pretraining sample

[`v4/tokenizer/tokenizer-32K.json`](v4/tokenizer/tokenizer-32K.json) is a byte-level BPE tokenizer trained on the downloaded sample. It uses individual-digit splitting, `add_prefix_space=False`, and a full byte alphabet. It has passed exact round-trip checks and has a maximum token ID of 31,999. It has not been formally declared final.

Special tokens in ID order are `<|im_start|>`, `<|im_end|>`, `<|pad|>`, `<|endoftext|>`, `<think>`, and `</think>`.

The intended pretraining mix by tokens is **70% FineWeb-Edu (`sample-10BT`), 15% Cosmopedia-v2, and 15% Stack-Edu (Python)**. Download budgets use upstream token counts or a character estimate, so the stored sample has different proportions under this tokenizer:

| Source | Documents | Actual tokens, including EOT | Share |
| --- | ---: | ---: | ---: |
| FineWeb-Edu | 84,377 | 91,265,189 | 72.129% |
| Cosmopedia-v2 | 23,319 | 17,262,193 | 13.643% |
| Stack-Edu Python | 21,524 | 18,003,105 | 14.228% |
| **Total** | **129,220** | **126,530,487** | **100%** |

The sample was downloaded with a 125M proxy-token budget. The `sample-100M` directory name is approximate. Source sampling to enforce 70/15/15 has not been implemented.

Tokenization processes each source separately in batches of up to 1,000 documents, appends one `<|endoftext|>` (ID 3) per document, and writes a flat binary stream with dtype `<u2` (little-endian uint16). No padding or truncation is configured. The three `.bin` files occupy 253,060,974 bytes, approximately 253 MB. `metadata.json` records token counts, the tokenizer path, and the EOT string.

On 2026-11-13, every saved token was compared against re-tokenizing every source document plus its EOT. All three streams matched exactly, file sizes matched metadata, all IDs were valid, and EOT counts matched document counts. First, middle, and last documents per source also decoded exactly to the original text, including Python whitespace.

There is no validation split yet. Packing, attention across document boundaries, and training-time source sampling remain to be implemented. EOT separators alone do not prevent attention across documents.

## Getting started

Development uses native Windows on an RTX 3090, Ryzen 5 5600X, and 64 GB RAM. [`v4/requirements.txt`](v4/requirements.txt) declares dependency pins, including PyTorch, `triton-windows` on Windows, FLA, Pydantic, Tokenizers, and Datasets. Install CUDA-enabled PyTorch and the platform-appropriate Triton first, then the remaining requirements. The pins describe the declared environment; compatibility still depends on the installed CUDA and platform setup.

Run V4 commands from inside `v4/` so local imports resolve:

```bash
git clone https://github.com/gavmc/Mini-K3.git
cd Mini-K3/v4
```

### Download the sample

```bash
python -m prep.download --total-tokens 125000000 --out-dir data/sample-100M
```

The downloader writes one parquet per source. Stack-Edu content is fetched from S3 using AWS credentials loaded from `v4/.env`; keep credentials private. Use `--sources` to select individual sources and `--help` for available options.

### Tokenize the sample

Create the output directory, then run the script's configured entry point:

```bash
python -c "from pathlib import Path; Path('data/tokenized/sample-100M').mkdir(parents=True, exist_ok=True)"
python -m tokenizer.tokenizer
```

This uses the existing 32K tokenizer and writes `fineweb-edu.bin`, `cosmopedia-v2.bin`, `stack-edu-Python.bin`, and `metadata.json` under `data/tokenized/sample-100M/`. Existing output files are overwritten. Raw data and tokenized datasets are excluded from git.

Read a binary stream without loading the whole file into RAM:

```python
import numpy as np

tokens = np.memmap(
    "data/tokenized/sample-100M/fineweb-edu.bin",
    dtype="<u2",
    mode="r",
)
```

The files have no header; supply the dtype when reading. Convert training batches to `torch.long` before passing them to the model.

### Run the V4 synthetic benchmark

```bash
python -m tests.throughput.throughput
```

Edit `runs` and `batch_sizes` at the top of [`throughput.py`](v4/tests/throughput/throughput.py). Run one model and one batch size per process: multiple configurations in one process previously hit the compiler recompile limit and fell back to eager execution. Results append to `v4/tests/throughput/throughput.md`.

The benchmark includes BF16 autocast, backward, gradient clipping, both optimizers, and both schedulers. It is a synthetic performance probe. A complete V4 training or generation command is not available yet.

## Custom KDA kernels

[`v4/kernels/`](v4/kernels/) contains educational PyTorch and Triton recurrent and chunkwise implementations and backward comparisons. These files currently import `dtypes`, `torch_map`, and `tl_map`, which no longer exist in `config.py`, so the kernel scripts do not run as-is. Model training uses FLA.

## Earlier checkpoint

An earlier model is available at [gav6/mini-k3-125m-sft](https://huggingface.co/gav6/mini-k3-125m-sft). It uses its own generation code rather than the standard Transformers loading interface.

That checkpoint is a research artifact. Instruction following, factual accuracy, repetition, and longer responses remain areas for improvement. Its results are separate from V4, which has not yet undergone pretraining.

## Next steps

- [x] Implement the dense KDA/MLA backbone, tied head, and explicit cache interfaces.
- [x] Implement Muon + AdamW and WSD scheduling.
- [x] Benchmark eight candidates and choose a shortlist.
- [x] Download, tokenize, and verify the pretraining sample.
- [ ] Add GPU observability to investigate system bottlenecks.
- [ ] Reserve validation documents and implement packing and source sampling.
- [ ] Verify causality, full-sequence versus cached decoding, and safe cache branching.
- [ ] Check finite losses/gradients, fixed-data overfitting, fused CE, and compiled/eager agreement.
- [ ] Build training, checkpointing, and resume, including RNG and data position.
- [ ] Run learning pilots and tune optimizer rates before the main run.
- [ ] Build evaluation, generation, and SFT; check copying, constraints, summarization, and stopping.
- [ ] Explore MoE and AttnRes as separate later ablations.

## References

See [`v4/sources.md`](v4/sources.md) for architecture papers, implementation references, training research, dataset cards, and tokenizer/storage documentation.

Built by [Gavin McLaughlan](https://github.com/gavmc).
