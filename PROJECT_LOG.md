# Mini-K3 project log

Chronological record of decisions, measurements, and research for Mini-K3. Oldest first. Append new dated entries at the bottom; do not rewrite old ones. The current state of the project lives in `AGENTS.md`.

Conventions used in entries:

- **Decision**: something Gavin chose.
- **Measured**: a result Gavin ran and reported. Unless an entry says otherwise, the agent read the output but did not run it.
- **Inspected**: the agent read the source. This is not evidence that it runs.
- **Proposal**: suggested by an assistant, not adopted or tested.

Entries before 2026-11-11 were reconstructed from the earlier single-file handoff and from git history, so some dates are commit dates rather than the day the work happened.

---

## 2026-08: v1 to v3

- v1 to v3 were built and remain in the repository for reference. Commits on 2026-08-06/07 added the v3 architecture with a dense FFN, FLA kernels for KDA, MoE, and AttnRes blocks.
- Earlier checkpoint published: https://huggingface.co/gav6/mini-k3-125m-sft. Its model card reports 125.9M total / 63.3M active parameters, 2.39B pretraining tokens, then SFT.
- Observed in its sample outputs: repetition, factual errors, instruction failures, poor long responses. The cause was never isolated; no single architecture bug was established.

## 2026-09-23: v4 planned

- **Decision**: start v4 with the goal of a useful small Kimi-inspired model trained from scratch on one RTX 3090.
- Initial target: roughly 120 to 155M total parameters with MoE, 2,048-token context. Million-token context is not a goal.
- Desired behavior: instruction following, rewriting, short summarization, copying names and numbers, list constraints, clean stopping.

## 2026-09-23 to 2026-10-12: custom Triton kernels

- Recurrent KDA forward and backward kernels written (`v4/kernels/recurrent_kda.py`), plus a PyTorch chunkwise reference (`v4/kernels/chunkwise_kda.py`).
- **Inspected**: recurrent backward consumes the real `[B,T+1,H,dk,dv]` state history, supports an optional final-state gradient, and returns five input gradients plus the initial-state gradient. It launches one program per batch/head over the whole value dimension, which avoids partial value-tile reductions. Forward uses value tiles of 32.
- **Inspected**: the chunkwise file is a PyTorch reference (FP32, triangular solves, handles short final chunks and empty sequences), not a Triton training kernel.
- **Inspected**: `v4/kernels/tests.py` compares manual and Triton backward against autograd at `B=1,T=256,H=8,dk=32,dv=64`, FP32, `rtol=atol=1e-4`. No passing run was recorded.
- Lessons noted while debugging the backward kernel: test against real recurrence history, not random history; value tiling creates partial reductions for dq/dk/dg/dbeta that must not race or be dropped; load pointers before arithmetic; BF16 output rounding can turn tiny FP32 differences into one-ULP differences.

## 2026-10-16: move to FLA for training

- **Decision**: use FLA kernels for the main training path. Keep the custom Triton work as educational reference code.
- Both kernel modules still import dtype names (`dtypes`, `torch_map`, `tl_map`) that no longer exist in `v4/config.py`, so they do not run as-is.

## 2026-10-21 to 2026-10-23: full v4 model

- MLA and SwiGLU FFN added (10-21). Full model with ordinary additive residuals ran (10-23). Explicit weight init added (10-23).
- Architecture as built: groups of three KDA mixers then one gated MLA, a dense SwiGLU FFN after every mixer, separate pre-attention and pre-FFN RMSNorms, final RMSNorm, tied 32K embedding/output weights, fused linear cross-entropy.
- Init recipe implemented: normal std 0.02 for all Linear and Embedding weights, RMSNorm gains one, and std `0.02/sqrt(2*layers)` for `KDA.o_proj`, `MLA.o_proj`, `FFN.down`. KDA decay parameters and convolution init are left alone. This is an experimental recipe, not shown to beat PyTorch defaults.
- **Inspected** at the time: the tied embedding/output parameter was initialized twice. Fixed later (see 2026-11-02).
- A `[1,5]` input producing `[1,5,32000]` logits was observed. That confirms shapes only.

## 2026-10-26 to 2026-10-31: optimizer and scheduler

- Educational Muon written (10-26): `newton_schulz`, `muon_update`, `CustomMuon`. Not used for training.
- `init_optim` added (10-29), returning PyTorch `Muon` and `AdamW`. Embeddings, convolutions, and vectors go to AdamW; other matrices go to Muon.
- `WSDScheduler` added (10-30): linear warmup, constant phase, `1 - sqrt(progress)` cooldown, zero at and after `total`.
- A CPU-only scheduler check reportedly passed (warmup/stable/cooldown values, multiple groups, zero-length phases, post-total clamp). That revision had phase validation; validation was removed afterwards. Not reproduced by an agent.

## Undated, before 2026-11: environment problem

- Gavin reported: Python 3.12 on native Windows, torch 2.5.1+cu124, Transformers 5.19.0, FLA 0.5.2, triton-windows 3.7.1.post27. Import failed because Transformers called `torch.accelerator`, which that torch lacks. Later AutoModel/GenerationMixin errors were wrappers around the same root error.
- **Proposal**: a fresh environment with a compatible Torch/Triton pair (Torch 2.12/2.13 with Triton 3.7 per the triton-windows table). Do not monkeypatch `torch.accelerator`. A successful reinstall was never explicitly reported, though benchmarks ran afterwards.
- W&B billing screenshot showed Training usage `$0 / $1000` with a $2.70/GPU-hour tooltip. It is unknown whether that is credit or a spending cap. Do not budget it as free pretraining compute.

## 2026-10 to 2026-11: scope and research discussions

Size and scope:

- **Decision**: widen candidates to roughly 150M, 300M, and 500M, with a dense FFN baseline before testing MoE.
- **Decision**: working main-run target of about 20B pretraining tokens, pilots additional unless budgeted otherwise.
- Eight candidates defined (analytical parameter estimates; all use `dk=dv=d_head=64`, heads `d_model/64`, `dc=d_model/4`):

| ID | Width | Pairs | Heads | FFN width | Approx. total M |
| --- | ---: | ---: | ---: | ---: | ---: |
| D150 A | 512 | 28 | 8 | 2304 | 151.3 |
| D150 B | 576 | 24 | 9 | 2304 | 152.6 |
| D150 C | 640 | 20 | 10 | 2560 | 158.4 |
| D300 A | 640 | 40 | 10 | 2560 | 296.3 |
| D300 B | 768 | 28 | 12 | 3072 | 302.2 |
| D300 C | 896 | 20 | 14 | 3584 | 298.3 |
| D500 A | 896 | 36 | 14 | 3584 | 514.1 |
| D500 B | 1024 | 28 | 16 | 4096 | 525.5 |

- **Proposal**: ablations, one at a time rather than a full grid: ordinary vs Block AttnRes; dense vs small MoE; pure KDA vs 3:1 vs 1:1 hybrid; latent width; value-head width; SwiGLU vs alternatives.
- **Proposal**: pilot order of correctness, then throughput/memory, then about 100M-token architecture pilots, then extend the strongest toward 500M tokens, then pick a main run. Short pilots can mis-rank long runs.

Research read (paraphrased; none of it is tested on Mini-K3):

- Kimi K3 §2.2 and the Attention Residuals paper: AttnRes is learned attention across depth, not a long-context mechanism. Table 2, in a 3:1 KDA/MLA MoE hybrid at 8,192 context:

| Active params excl. embeddings | Baseline loss | Block AttnRes | Full AttnRes |
| --- | ---: | ---: | ---: |
| 194M | 1.931 | 1.909 | 1.899 |
| 296M | 1.829 | 1.809 | 1.804 |
| 528M | 1.719 | 1.693 | 1.692 |

  Token budgets were 38.7B, 62.1B, 119B. Those are active MoE sizes, not total dense sizes. The paper's fitted 1.25x compute advantage is not a measured 3090 speedup. If implemented, initialize pseudo-queries to zero as the paper specifies.
- MobileLLM: supports trying deeper, thinner sub-billion dense transformers. Does not show that narrower is better for a KDA hybrid.
- Xiong et al.: pre-norm is a sound optimization baseline. Not a reason to drop warmup.
- SmolLM2: relevant for small-model data and recipe design. MiniLLM: distillation is another possible route, not implemented.
- LR schedules. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*: constant LR plus cooldown can match cosine; `1 - sqrt(progress)` beat linear cooldown; benefit levelled off around 20% cooldown; the best constant-LR peak was about half the cosine peak in the cited comparison. MiniCPM: WSD gives reusable pre-cooldown checkpoints. WSqD (July 2026): inverse-square-root stable phase, a later candidate. *Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning*: constant-LR checkpoints fine-tuned better in its 1B/8B experiments.
- Initialization: `nn.Linear` defaults are uniform `+/-1/sqrt(fan_in)`. Fixed std 0.02 changes narrow bottlenecks strongly. PyTorch defaults remain a legitimate baseline to compare against.
- Decay init: `dt_bias` uses an inverse softplus of log-uniform dt in [0.001, 0.1], but the gate is a sigmoid. With `a_log=0` and `z=0`, `g = -5*(1-exp(-dt))`, about -0.005 to -0.476, so retention starts around 0.995 to 0.621. dt is therefore not the initialized log-decay magnitude.
- Muon scaling: PyTorch `Muon` with `adjust_lr_fn=None` multiplies LR by `sqrt(max(1, A/B))` for an `[A,B]` matrix; `"match_rms_adamw"` uses `0.2*sqrt(max(A,B))`. These are different recipes, and equal numeric Muon/AdamW LRs do not mean equal update sizes.

Proposals from those discussions (not adopted):

- Main-run schedule: linear warmup over about 0.5% of tokens (100M of 20B), hold separately tuned peaks to 16B, then `1 - sqrt(progress)` cooldown over the final 20% to zero. A 1% LR floor is an untested alternative.
- Schedule comparison: tune peaks first; compare warmup+cosine against WSD at equal tokens; save a full checkpoint before cooldown and branch cooldown variants from it; keep pre-cooldown checkpoints for extensions; compare stable vs cooled checkpoints after the same small SFT.
- Scheduler guards: integer phases, `total > 0`, `warmup + cooldown <= total`, finite nonnegative peak.

Sources consulted:

- Kimi K3: https://arxiv.org/html/2607.24653v2
- Attention Residuals: https://arxiv.org/abs/2603.15031 and https://github.com/MoonshotAI/Attention-Residuals
- Kimi Linear: https://arxiv.org/abs/2510.26692
- MobileLLM: https://arxiv.org/abs/2402.14905
- SmolLM2: https://arxiv.org/abs/2502.02737
- MiniLLM: https://arxiv.org/abs/2306.08543
- Pre-norm (Xiong et al.): https://arxiv.org/abs/2002.04745
- FLA: https://github.com/fla-org/flash-linear-attention
- GPT-2 residual init: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
- PyTorch Linear init: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
- PyTorch Muon: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- CUDA timing/memory: https://docs.pytorch.org/docs/stable/notes/cuda.html
- AMP: https://docs.pytorch.org/docs/stable/amp.html
- torch.compile tutorial: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- triton-windows: https://github.com/triton-lang/triton-windows
- PyTorch versions: https://pytorch.org/get-started/previous-versions/
- W&B pricing: https://site.wandb.ai/pricing/ and https://www.coreweave.com/forge-pricing/post-training
- LR cooldown comparisons: https://arxiv.org/html/2405.18392v3
- MiniCPM / WSD: https://arxiv.org/html/2404.06395v3
- WSqD: https://arxiv.org/html/2607.10959v1
- Schedule and downstream SFT: https://arxiv.org/html/2603.16127v1
- PyTorch scheduler source: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py

## 2026-11-02: optimizer fixes, W&B smoke test, throughput sweeps, shortlist

Fixes and settings:

- `OptimConfig` defaults set to Muon LR 0.01, AdamW LR 0.001, momentum 0.95, betas (0.9, 0.95), clip 1.0, zero decay. Provisional starting values, not tuned.
- `init_optim` changed to ordered lists, skips the tied `output` weight, and puts `_no_weight_decay` vectors in a no-decay AdamW group.
- Tied-weight double initialization fixed with an identity check in `init_weights`.
- Gavin reported completing a config check and correctness checks. Their outputs were not inspected by an agent.
- **Decision**: work order of optimizer LR settings, then throughput/memory sweep, then data and learning experiments.

W&B smoke test (local run `w4uci0jl`):

- **Measured**: 50 updates on seven next-token targets, loss 10.3599 to 0.5444. Shows the model can fit one tiny sequence and that logging works. The run stayed inside scheduler warmup; printed gradient norms were pre-clipping.

Earlier laptop numbers (RTX 4060 Laptop GPU, B=1, T=2048, older 25-step script): 1864.9, 1409.4, 2920.6 tokens/sec. Too noisy to rank anything.

Eager full sweep (desktop, 5 warmup + measured updates, T=2048, BF16 autocast, Muon + AdamW, synthetic tokens; saved in `v4/tests/throughput/throughput-non-compile.md`). Fastest successful batch per candidate:

| Candidate | Batch | Tokens/sec | Peak allocated GiB | Peak reserved GiB | Ideal 20B-token days |
| --- | ---: | ---: | ---: | ---: | ---: |
| D150_A | 8 | 18,952.7 | 22.88 | 23.23 | 12.2 |
| D150_B | 8 | 21,498.5 | 21.34 | 21.71 | 10.8 |
| D150_C | 8 | 23,084.0 | 19.98 | 20.43 | 10.0 |
| D300_A | 4 | 9,561.0 | 21.11 | 21.36 | 24.2 |
| D300_B | 4 | 11,051.5 | 18.39 | 18.62 | 20.9 |
| D300_C | 4 | 13,023.7 | 15.78 | 16.18 | 17.8 |
| D500_A | 2 | 5,391.8 | 16.58 | 18.45 | 42.9 |
| D500_B | 2 | 5,790.9 | 15.29 | 16.32 | 40.0 |

- D150 ran at batches 1, 2, 4, 8. D300 ran at 1, 2, 4 and hit OOM at 8. D500 ran at 1, 2 and hit OOM at 4 and 8.
- Within each size, the wider/shallower candidate was faster and used less memory.
- An outside reviewer worried the sweep used a fixed `dc=144`. It did not: every JSON entry sets `dc=d_model//4`.

First compiled sweep (`throughput-compiled-1.md`):

- All combinations ran in one process and hit `recompile_limit (8)` warnings, so later results likely include eager fallback. Not a fair comparison. Its D300_C batch 8 OOM was overturned by the follow-up below.

Compiled follow-up, one model and batch per process (`throughput.md`):

| Candidate | Actual parameters | Batch | Tokens/sec | Peak allocated GiB | Peak reserved GiB | Ideal 20B-token days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D150_B | 152,893,890 | 8 | 23,389.4 | 16.75 | 17.13 | 9.9 |
| D150_B | 152,893,890 | 10 | 25,364.6 | 20.92 | 21.83 | 9.1 |
| D150_C | 158,661,750 | 8 | 25,087.4 | 15.73 | 16.16 | 9.2 |
| D150_C | 158,661,750 | 10 | 26,829.3 | 19.78 | 20.24 | 8.6 |
| D300_C | 298,734,066 | 4 | 13,783.7 | 12.81 | 13.31 | 16.8 |
| D300_C | 298,734,066 | 8 | 17,258.9 | 22.59 | 22.95 | 13.4 |

- D150_B and D150_C both hit OOM at batch 12. Batch 11 was not tried.
- D150_C batch 10 is about 5.8% faster than D150_B batch 10 and uses 1.14 GiB less. D300_C batch 8 needs about 55% more time than D150_C batch 10 for the same tokens.
- "Ideal days" is `20e9 / tokens_per_second / 86400`: synthetic-loop time only, excluding data loading, evaluation, checkpointing, and downtime.
- The output files do not record GPU, environment, resolved configs, or compiler logs.

- **Decision**: main-run shortlist is **D150_B, D150_C, D300_C**. No final model or recipe chosen.

## 2026-11-05: refactor and first tokenizer helper (commit `4f926bd`)

- Code moved into `v4/model/` (`model.py`, `layers.py`) and `v4/optimization/` (`optim.py`, `scheduler.py`). Benchmark and result files moved to `v4/tests/throughput/`. `v4/tests.py` removed.
- `check_config` added in `v4/util.py` and called at the start of `MiniK3.__init__`: asserts `layers % 4 == 0` and `d_head == d_model / heads`.
- Benchmark config path fixed to be independent of the working directory. Results now append to `throughput.md` instead of overwriting. The earlier finite-loss checks are no longer in the script.
- First `train_tokenizer` helper: plain BPE with `[UNK]`, Whitespace pre-tokenization, and BERT-style special tokens. Replaced on 2026-11-10.
- **Inspected** only: syntax parsing passed; the config-loading expression loaded all eight candidates. Nothing was run on GPU.

## 2026-11-10: byte-level tokenizer and dataset start (commits `6aa8296`, `37225eb`)

- `v4/tokenizer.py` rewritten as byte-level BPE: ByteLevel pre-tokenizer (`add_prefix_space=True`), ByteLevel decoder and post-processor, full byte initial alphabet, `min_frequency=2`. Special tokens in order: `<|im_start|>`, `<|im_end|>`, `<|pad|>`, `<|endoftext|>`, `<think>`, `</think>`.
- `v4/data/download.py` started: opens streaming handles to `HuggingFaceFW/fineweb-edu`, `HuggingFaceTB/smollm-corpus` (cosmopedia-v2), and `HuggingFaceTB/stack-edu` (python), plus an S3 helper to fetch stack-edu file contents. `tokenizer_sample` does not yet sample or write anything.
- `datasets` and `boto3` added to `v4/requirements.txt`.
- No tokenizer has been trained yet.

## 2026-11-11: documentation split and code review

- The single handoff file was split: `AGENTS.md` now holds only current state and instructions; this file holds history. 

- **Inspected** by Claude: `layers.py`, `model.py`, `optim.py`, `scheduler.py`, `config.py`, `util.py`, `throughput.py`, `tokenizer.py`, `data/download.py`. No code was changed or run. No correctness bug was found in the forward path, masking, tying, or schedule arithmetic. Findings are listed under "Known issues" in `AGENTS.md`; the main ones:
  - The scheduler writes `cfg.base_lr` to every group and ignores the optimizer's LR. The default `SchedulerConfig.base_lr=0.005` would silently give both optimizers the wrong LR if not overridden.
  - Muon runs with zero weight decay and no warmup in the only loop that exists. Residual-output weights start very small, so warmup matters for Muon; decay for a long run is untested.
  - RMSNorm gains sit in the decayed AdamW group; harmless only while decay is zero.
  - MLA has no positional signal of its own (NoPE by design) and depends on the KDA layers before it.
- **Proposals** from the review (not adopted): add a same-size plain-attention baseline to the pilots; make the scheduler a multiplier on the optimizer's own LR; build the data path and correctness gates before more throughput work.

## 2026-11-11 (later): review fixes and next-step plan

- Gavin fixed several review items (uncommitted at the time of writing): RMSNorm gains moved to the zero-decay AdamW group; `SchedulerConfig.base_lr` default changed from 0.005 to `None`; the redundant `initialized.add` removed; `MoE.forward` given `self`; `wandb/` and `PROJECT_LOG.md` added to `.gitignore`.
- **Decision**: pretraining mix by tokens is 70% fineweb-edu, 15% cosmopedia-v2, 15% stack-edu (python).
- **Decision**: next steps are to finish the tokenizer, build a pretraining sample of about 100M tokens, then add GPU observability to find system bottlenecks.
- **Proposals** from Claude (not adopted): pull the sample by streaming with a shuffle buffer, sized by each dataset's own token-count column, then trim to the exact mix after tokenizing with the real tokenizer; store tokens as a flat uint16 file with a document-level validation split; start weight-decay pilots at Muon 0.1 and AdamW 0; make `base_lr` a required field.

## 2026-11-11 (later still): download script

- Gavin wrote `v4/prep/download.py` (replacing `v4/data/download.py`) and added `DataConfig`. `OptimConfig.muon_weight_decay` set to 0.1; `SchedulerConfig.base_lr` made a required field.
- **Measured** (read by Claude from the output files): a `total_tokens=10_000` test wrote `sample/fineweb-edu.parquet` (10 rows, 7,364 proxy tokens) and `sample/cosmopedia-v2.parquet` (3 rows, 2,013 proxy tokens). No stack-edu file exists; why is not known.
- **Inspected** by Claude: `_fetch` swallows all exceptions, so a broken S3 setup would make the stack-edu step run without ending or reporting an error.
- **Observed** by Claude on the running stack-edu step (started 13:20): one CPU core busy, 8.6 GB RAM, zero established network connections, no AWS credentials on the machine. Every S3 fetch was failing locally and being skipped by `_fetch`, so the budget could never be reached.

## 2026-11-11 (evening): S3 working, 100M-token run prepared

- Gavin added AWS credentials (IAM user with `AmazonS3ReadOnlyAccess`, loaded from `v4/.env`) and reports a small test run of all three sources worked.
- `_fetch` now lets S3 errors other than a missing file propagate. `bytes_per_token` changed to 3.0. Output directory set to `data/sample-100M`, `total_tokens=100_000_000`.
- **Inspected** by Claude before the 100M run: `v4/data/sample-100M/` is not covered by `.gitignore`. Suggested also saving document IDs, so the validation set can be kept out of the later full-size download.

## 2026-11-11 (late): 100M sample downloaded, first tokenizers

- Gavin pulled the sample with a 125M proxy-token budget into `v4/data/sample-100M/`: fineweb-edu 237 MB, cosmopedia-v2 53 MB, stack-edu-Python 24 MB (parquet). `download.py` now takes command-line options, including `--sources`.
- Gavin trained byte-level BPE tokenizers at 16K, 32K, 64K, 128K on the full sample (`v4/tokenizer/`), with `add_prefix_space=True` and no digit splitting.
- **Measured** by Claude on 2,000 training documents per source (characters per token; training data, so slightly optimistic):

| Vocab | fineweb-edu | cosmopedia-v2 | stack-edu-Python |
| --- | ---: | ---: | ---: |
| 16K | 4.25 | 4.74 | 2.99 |
| 32K | 4.55 | 5.11 | 3.22 |
| 64K | 4.77 | 5.32 | 3.39 |
| 128K | 4.90 | 5.43 | 3.51 |

- Round trip: every document decodes with an extra leading space; a space is also inserted after each special token. Special-token IDs are 0-5 and each encodes as one token.
- **Proposal** (Claude): keep 32K; retrain with `add_prefix_space=False` and individual-digit splitting. Reason: each doubling past 32K improves compression only 3-5% while the tied embedding grows by `vocab x d_model` parameters (64K adds about 20M at width 640) and output-layer compute grows with it.
- Retrained all four sizes with `add_prefix_space=False` (Gavin's run of Claude's check script): 2,000/2,000 documents per source round-trip exactly; compression unchanged to two decimals. Individual-digit splitting was not added.
- Retrained 32K with individual-digit splitting. **Measured** by Claude: exact round trip 2,000/2,000 per source; zero multi-digit tokens; characters per token 4.41 / 5.08 / 3.14 (fineweb / cosmopedia / stack-edu), down from 4.55 / 5.11 / 3.22. The other sizes were deleted.

## 2026-11-13: sample tokenized as uint16 and fully verified

- Gavin completed tokenization of all three downloaded parquet files using `v4/tokenizer/tokenizer-32K.json` and `tokenize_dataset(tok_file, files, path)` in `v4/tokenizer/tokenizer.py`.
- **Inspected by Codex**: each source is loaded separately, with batches of up to 1,000 text documents. `encode_batch(..., add_special_tokens=False)` produces IDs; one `<|endoftext|>` (ID 3) is appended per document. IDs are concatenated without padding or configured truncation and written with NumPy dtype `<u2` (little-endian uint16).
- Outputs are `fineweb-edu.bin`, `cosmopedia-v2.bin`, `stack-edu-Python.bin`, and `metadata.json` under `v4/data/tokenized/sample-100M/`. These are raw binary arrays without headers. Metadata stores token counts in that source order, the tokenizer path, and the EOT string. The script's main entry point can be run with `python -m tokenizer.tokenizer` from `v4`; the output directory must already exist and existing outputs are overwritten.

**Measured by Codex from the files and source parquet rows**, including appended EOT tokens:

| Source | Documents | Tokens | Bytes | Actual share |
| --- | ---: | ---: | ---: | ---: |
| fineweb-edu | 84,377 | 91,265,189 | 182,530,378 | 72.129% |
| cosmopedia-v2 | 23,319 | 17,262,193 | 34,524,386 | 13.643% |
| stack-edu-Python | 21,524 | 18,003,105 | 36,006,210 | 14.228% |
| Total | 129,220 | 126,530,487 | 253,060,974 | 100% |

- **Verification executed by Codex**: re-tokenized every source document and compared every saved ID, including each appended EOT, against its source-specific binary stream. All three matched exactly with no missing or trailing tokens. Each file size equals its metadata token count times two. Every stored ID belongs to the tokenizer vocabulary; tokenizer maximum ID is 31,999, safely within uint16. EOT counts equal document counts; no source encoding contained a literal EOT token.
- **Decode checks executed by Codex**: the first, middle, and last document of each source decoded exactly to its original text with `skip_special_tokens=False`, including Python whitespace. This was nine decoded samples, separate from the full token-by-token comparison above.
- The existing 70/15/15 target remains Gavin's decision. Actual stored shares differ because the download used proxy counts from other tokenizers or a character estimate. No trimming or source sampler was added.
- **Recommendations from Codex, not implemented**: enforce the target mix by sampling equal-length sequences from the separate source files; reserve validation documents before training; keep packing in the loader. No validation split, training loader, or attention-boundary policy exists yet. EOT separation is implemented in storage but does not itself mask attention across documents.
- Remaining metadata limitations: no dtype field, tokenizer hash, source revision, document offsets, or split information. No pre-cast ID-range check exists in the tokenization script, although the current tokenizer and saved output passed range verification. Tokenizer finalization was not declared, and model/training correctness gates were not exercised by these data checks.

## 2026-11-13: README and source references refreshed

- Updated `README.md` against the current V4 model, optimizer/scheduler, benchmark results, tokenizer, and data preparation source. Removed obsolete placeholder architecture descriptions, proposed model sizes, broken layer paths, and instructions presenting the reference kernel tests as runnable. Added the measured shortlist, verified token counts, current commands, uint16 format, and remaining training work.
- Organized `v4/sources.md` by architecture, small-model research, optimization/schedules, datasets, and tokenization/storage. Retained the original reading list and added official dataset/API references and WSD/cooldown papers. Distinguished implemented behavior, measurements, targets, and later experiments.
- Checked local Markdown link targets and code fences; no training or data files were changed.

## 2026-11-18: pilot training script and first 100M-token run (D150_C)

Script:

- **Decision**: load data with a plain `get_batch` function over `np.memmap`, rather than Hugging Face `datasets` or a PyTorch `Dataset`/`DataLoader`. The reasons discussed: the data is already tokenized and verified, about 41 KB per micro-batch is trivial load, resume needs only the update number, and every model sees identical batches.
- Gavin wrote `v4/tests/runs/sample_train.py`, with help from Claude. 40 sequences per update, as 4 micro-batches of 10 with gradient accumulation (40 divides by both 10 and 8, so D300_C can match tokens per update). 1,220 updates for 100M tokens. BF16 autocast, compiled. Validation split: the end of each source file, cut after an EOT, 240 windows per source, built by `split_source`. W&B logging of loss, pre-clip grad norm, both LRs, tokens seen, and per-source validation loss every 100 updates. Final weights saved to `v4/tests/runs/D150_C_100M.pt`.
- Schedule as run: warmup 6 updates (0.5% of 1,220, carried over from the 20B proposal), cooldown 244 (20%), total 1,220. Default `OptimConfig` (Muon 0.01, AdamW 0.001, Muon WD 0.1). Data seed 0.
- `python -m tests.runs...` first failed because `ultralytics` and `playwright_stealth` ship a top-level `tests` package in site-packages, which shadowed `v4/tests`. Fixed by adding `v4/tests/__init__.py`.
- The run's recorded git commit is `ed45096`, but `sample_train.py` was uncommitted, so that commit does not contain the code that ran.

**Measured** (W&B run `wg617mfb`, "deep-armadillo-1"; history pulled and analysed by Claude):

| Update | val mix | fineweb-edu | cosmopedia-v2 | stack-edu-Python |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10.492 | 10.496 | 10.475 | 10.488 |
| 100 | 5.539 | 5.705 | 5.484 | 4.820 |
| 300 | 4.508 | 4.749 | 4.187 | 3.703 |
| 500 | 3.922 | 4.245 | 3.638 | 2.701 |
| 800 | 3.655 | 3.999 | 3.323 | 2.380 |
| 1000 | 3.537 | 3.887 | 3.190 | 2.253 |
| 1220 (end) | 3.346 | 3.704 | 2.961 | 2.057 |

- Initial validation loss 10.49 matches the expected value for this init. Logit std is about 0.02 × sqrt(640) ≈ 0.51, so loss ≈ ln(32000) + 0.51²/2 ≈ 10.50.
- No non-finite loss or gradient norm. The pre-clip grad norm was above 1.0 (clipping active) for the first ~45 updates, with a peak of 12.5 at update 14 just after the LR reached its peak. After that it fell steadily: about 0.70 (updates 50-99), 0.38 (300s), 0.32 (800s), 0.24 (cooldown).
- Train-loss jumps of 0.15-0.35 occur throughout, without matching grad-norm jumps. The likely cause is that batches differ in source mix, since stack-edu loss is far below fineweb loss. Inferred, not checked per batch.
- The cooldown gave a large gain: val mix fell 0.19 from update 1000 to 1220, against 0.04-0.08 per 100 updates late in the constant phase. A guess, not tested: the constant LR may be limiting progress, which makes peak-LR tuning worthwhile.
- Train/val gap (train averaged over ±10 updates): +0.01 to +0.03 at updates 100-200, growing to +0.10 to +0.16 after update 800. This fits windows being sampled with replacement, which makes repeats increasingly common (about 0.77-0.89 epochs per source by the end). Inferred, not proven.
- Throughput: 2.71 s per update = about 30,200 tokens/s in steady state, faster than the 26,829 tokens/s benchmark at batch 10. The likely reason is one clip and optimizer step per 4 micro-batches instead of one per batch. An evaluation pass took about 14 s; 14 evaluations add about 3 minutes. Time to first update was 230 s (two compiles plus the first evaluation). Total runtime 3,575 s.
- System (W&B system metrics, after the first 10% of the run): GPU utilisation averaged 96%, with dips during evaluation. GPU memory in use was about 23.8 GB of 25.8 GB (all processes). Host CPU was about 14%, and the process used about 2.7 GB RAM. Data loading was not a bottleneck.
- At 30,200 tokens/s, 20B tokens would take about 7.7 ideal days for D150_C, excluding evaluation and downtime.

## 2026-11-21 Pilot training configuration

The pilot now reads sampling, batch sizes, optimizer settings, and schedule lengths from `TrainConfig`. The default warmup is 100 updates. The November 18 run used six.
