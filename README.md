# MiniK3 — Scaled-Down Kimi K3 Reproduction

A single-GPU PyTorch reproduction of the Kimi K3 architecture (Moonshot AI, 2026), scaled to ~155M total / ~70M active parameters so it trains on an RTX 3090/4090. This is the **v3** iteration of the repo; see [`v1/`](v1/) and [`v2/`](v2/) for earlier experiments.

The goal is architectural fidelity — every component (KDA, Gated MLA, Block AttnRes, Stable LatentMoE with SiTU-GLU) is implemented to match the equations in the K3 technical report, just shrunk to fit consumer VRAM.

> **Status:** Architecture is implemented and benchable. Next up: Quantile Balancing, Per-Head Muon optimizer, and a real training run.

---

## Architecture

| Component | MiniK3 (v3) | K3 (report) |
|---|---|---|
| `d_model` | **640** | 7,168 |
| Latent MoE dim `d_latent` | **320** (0.5×) | 3,584 (0.5×) |
| Expert hidden `d_expert` | **512** | 3,072 |
| Shared expert hidden `d_shared` | **1024** | — |
| Layers | **12** | 93 |
| Attention | **9 KDA + 3 Gated MLA** | 69 KDA + 24 MLA |
| Routed experts | **16** | 896 |
| Active per token | **2** of 16 (8× sparsity) | 16 of 896 (56×) |
| Shared experts | **1** | 2 |
| Heads | **8** | 96 |
| KDA `d_k` / `d_v` | **128 / 64** | — |
| MLA `d_head` | **128** | — |
| Vocab | **16,000** (BPE, tied) | 160K |
| Context (bench) | **16K** | 1M |
| **Total / Active** | **~155.7M / ~69.7M** | 2.78T / 104B |

### Hybrid Attention (3:1 KDA : MLA)

- **KDA layers** use FLA's `chunk_kda` kernel with `use_qk_l2norm_in_kernel=True`. Inputs go through `ShortConvolution` (default `silu` / Swish), then L2-normalized queries/keys. Decay is lower-bounded via `g = g_min · sigmoid(exp(A_h) · z)` with `g_min = -5` and per-head log-scale `A_h` initialized to 0 (Eq. 5 in the report).
- **Gated MLA layers** compress KV into a latent `c = W_c x`, reconstruct k/v from `c`, and apply a full-rank sigmoid output gate. No positional encoding anywhere (NoPE) — the KDA layers handle position implicitly.
- The final layer is always MLA, matching K3's "additional Gated MLA layer at the end of the backbone."

### Block Attention Residuals (AttnRes)

Replaces the standard residual stream with attention over prior layer/module outputs. Each module (mixer or FFN/MoE) has its own learnable pseudo-query `w`, and the backbone accumulates partial sums within blocks. At block boundaries the full block representation becomes a source for subsequent layers. A final AttnRes aggregates all blocks for the head.

*Note:* The current implementation applies AttnRes at **module granularity** (2 per layer-pair) rather than K3's layer granularity, giving more pseudo-queries and finer-grained partial sums. `attnres_block=6` counts module outputs (~3 attention+FFN layers per block at L=12).

### Stable LatentMoE

- **Routed path:** `W_down` projects to latent space, then top-k experts (SiTU-GLU with `β₁=4`, `β₂=25`) operate in that latent space. Aggregated outputs pass through an RMSNorm before `W_up` projects back to full width — this is the "Normalized LatentMoE" from §2.3.1.
- **Shared expert:** one dense FFN (SiTU-GLU) that bypasses the latent path entirely. Layer 0 is a dense FFN instead of MoE, matching K3's single dense layer.
- **Router:** `sigmoid(W_r x)` at full width. Top-k selection uses a per-expert bias `b`, but mixture weights `p` are computed from the **unbiased** scores (Eq. 13). Currently uses sign-based load balancing (`bias_lr * sign(err)`). Quantile Balancing (Eq. 14 / Appendix C) is planned.
- **Capacity capping:** tokens beyond `1.5×` the target load per expert are zero-weighted (dropped). This is a training-time safety; K3 uses MoonEP's perfect balance instead.

### Dense FFN

Layer 0 uses a dense FFN with SiTU-GLU. All other layers use MoE.

---

## Project Structure

```
MoE/
├── README.md
├── v1/                     # GPT-2 style domain-specific MoE experts (legacy)
├── v2/                     # Earlier Kimi K3 scratch attempt, design notes in structure.md
└── v3/                     # Current MiniK3 implementation
    ├── model.py            # MiniK3 model (KDA, MLA, MoE, AttnRes)
    ├── config.py           # Model + training hyperparameters
    └── bench.py            # Benchmark: param counts, tok/s, MFU, token-budget projection
```

---

## Usage

### Benchmark

```bash
cd v3
python bench.py
```

This compiles the model (`torch.compile`), runs a few warmup + measured steps, and prints:
- Total / non-embedding / active parameter counts
- Step time and tok/s
- Model FLOP/s and MFU (vs RTX 3090 BF16 peak ~71 TFLOP/s)
- Chinchilla-optimal token budget and wall-clock estimates for 5B–30B tokens

Default config targets a 3090/4090 with compile + gradient checkpointing. At `B=1, T=16384` it should fit comfortably in 24 GB.

### Config

Edit `v3/config.py`:

| Field | Default | Notes |
|---|---|---|
| `d_model` | 640 | Hidden dimension |
| `layers` | 12 | Total attention+FFN pairs |
| `mix_ratio` | 4 | MLA every N layers (3:1 KDA:MLA) |
| `d_latent` | 320 | Latent space for routed experts (0.5× d_model) |
| `d_expert` | 512 | Expert hidden width |
| `d_shared` | 1024 | Shared expert hidden width |
| `n_experts` | 16 | Routed expert pool |
| `top_k` | 2 | Experts active per token |
| `vocab_size` | 16000 | Tied embeddings |
| `use_ckpt` | True | Gradient checkpointing (trades compute for memory) |

---

## What's Implemented vs Planned

| Component | Status |
|---|---|
| KDA (FLA `chunk_kda`, lower-bounded decay, full-rank sigmoid gate) | ✅ |
| Gated MLA (NoPE, full-rank sigmoid gate) | ✅ |
| Block AttnRes (per-module pseudo-queries, partial sums, final aggregation) | ✅ |
| Stable LatentMoE (latent routing, SiTU-GLU, normalized aggregate, shared expert) | ✅ |
| Dense first layer | ✅ |
| Sign-based load balancing | ✅ |
| Quantile Balancing (Eq. 14, exact `torch.quantile` version) | 🔄 Planned |
| 2nd shared expert per layer (K3 uses Ns=2) | 🔄 Under consideration |
| MTP layer | ❌ Skipped (inference only; small model) |
| Vision backbone (MoonViT-V2) | ❌ Skipped (text-only for now) |
| Per-Head Muon optimizer | 🔄 Next up |
| Weight clipping (K2-style) | 🔄 With Muon |
| Cosine LR, 1% warmup, wd=0.1 | 🔄 With Muon |
| Real data training loop | 🔄 Pending |

### Why skip the 2nd shared expert?

K3 uses 2 full-width shared experts per layer; we currently have 1. At our scale (12 layers, 8× sparsity) the routed experts are already well-trained (12.5% token coverage each), and a 2nd shared expert would add ~22M active params (+31% FLOPs) for unproven benefit on a single GPU. We're keeping it as a clean future A/B experiment.

### Why module-granularity AttnRes?

K3 applies AttnRes at the layer-pair level (mixer+FFN as one unit) with ~8 blocks of 12 layers. At L=12, strict layer-granularity would collapse to 1–2 blocks, losing the block structure entirely. Module-granularity gives us 4 blocks of 3 layers, which preserves the inter/intra-block dynamics at small depth.

---

## Hardware Target

- **Primary:** RTX 3090 (24 GB) or RTX 4090 (24 GB)
- **Precision:** `torch.bfloat16` autocast, compiled (`torch.compile`), gradient checkpointing
- **Expected throughput:** depends on sequence length; bench.py will report your actual tok/s and MFU

---

## References

- Kimi K3: Open Frontier Intelligence — Moonshot AI technical report, arXiv:2607.24653v1 (July 2026)
- Kimi Linear: An Expressive, Efficient Attention Architecture — arXiv 2510.26692
- flash-linear-attention — [`github.com/fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
- Muon — [kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)
