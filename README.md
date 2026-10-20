# Mini-K3

**Small language models built and trained from scratch on a single RTX 3090.**

Mini-K3 is an independent PyTorch research project exploring Kimi-inspired architectures at consumer-GPU scale: Kimi Delta Attention (KDA), gated multi-head latent attention, mixture-of-experts FFNs, and attention residuals.

The project has evolved from architectural reproduction toward a more practical question: **which of these ideas actually improve a small model under a limited training budget?** V4 is the current development effort. Earlier versions remain in the repository for reference.

> **Status:** V4 is under construction. Attention layers and experimental KDA kernels are present, but the full V4 language model and training pipeline are not yet complete. This is an educational research project, not an official Moonshot AI implementation or a production assistant.

## Goals

- Train a useful small language model on one 24 GB GPU.
- Improve instruction following, short-form generation, summarization, and rewriting over earlier checkpoints.
- Compare model width, depth, and parameter count through controlled experiments.
- Evaluate dense FFNs against small MoE configurations rather than assume MoE is better at this scale.
- Use established kernels for model training while retaining custom Triton implementations for learning and correctness experiments.

## Versions

| Version | Purpose | What is in the repository |
| --- | --- | --- |
| [`v1/`](v1/) | Early MoE experiments | Expert training and merging code |
| [`v2/`](v2/) | Earlier end-to-end Mini-K3 iteration | Model, data preparation, pretraining, SFT, sampling, optimizer code, and design notes |
| [`v3/`](v3/) | FLA-based architecture prototype | Hybrid KDA/MLA model, latent MoE, AttnRes, configuration, and a synthetic training benchmark |
| [`v4/`](v4/) | Current rebuild | Attention layers, dtype configuration, custom KDA experiments, and an unfinished model scaffold |

These versions are separate experiments. Their configurations, checkpoints, and entry points are not interchangeable.

## V4 architecture

### Kimi Delta Attention

The KDA layer in [`v4/layers.py`](v4/layers.py) includes:

- Separate query, key, and value projections.
- Short causal convolutions on Q, K, and V.
- Per-head update strengths and per-key-channel decay gates.
- Query/key L2 normalization inside the FLA kernel.
- Per-head output RMSNorm and a learned sigmoid output gate.
- Recurrent-state and convolution-state cache interfaces.

The layer uses [flash-linear-attention](https://github.com/fla-org/flash-linear-attention): `chunk_kda` for training and multi-token processing, and `fused_recurrent_kda` for single-token evaluation with gradients disabled.

Each KDA layer maintains its own recurrent state. During prefill it produces an output for every input token; during decoding it processes the new token using that layer's cached state.

### Gated latent attention

The `MLA` layer compresses each token to a normalized latent representation, reconstructs per-head keys and values, and applies causal scaled dot-product attention. A learned sigmoid gate modulates the attention output before the output projection.

It currently uses **no explicit positional encoding** and provides a latent-cache interface for incremental decoding, including an offset causal mask when extending a cache with multiple tokens.

This is a straightforward latent-attention implementation: it reconstructs keys and values from the cached latents on each call. It does not yet implement an optimized weight-absorbed MLA decoding path.

### Backbone and FFNs

The V4 backbone is not assembled yet: `MiniK3.forward` is a placeholder and the V4 `MoE` class is a scaffold. Layer scheduling, residual connections, FFNs, embedding/head tying, and model-wide cache management still need to be integrated.

The initial experiment plan is to establish a dense-FFN hybrid baseline, then test MoE and additional architectural features separately. A 3:1 KDA-to-MLA schedule is a starting candidate, not a demonstrated optimum at this size.

## Model-size experiments

V4's final size is not fixed. The following are **proposed configurations**, not trained models or the current defaults in `v4/config.py`.

| Candidate | Hidden width | Layers | Attention heads | SwiGLU intermediate width | Estimated total parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small baseline | 576 | 24 | 9 | 2,304 | ~153M |
| Main candidate | 768 | 28 | 12 | 3,072 | ~302M |
| Larger candidate | 896 | 36 | 14 | 3,584 | ~514M |

These estimates assume a 32K vocabulary, tied embeddings, 64-dimensional key/value heads, dense FFNs, and a 3:1 KDA/MLA schedule. Counts must be checked against the assembled models. Whether a configuration fits and trains efficiently depends on sequence length, batch size, checkpointing, and kernel behavior.

The working target is approximately **20B pretraining tokens for the selected model**, after smaller pilot runs. Architecture and training budgets may change based on measured throughput and validation results.

## Custom KDA kernels

[`v4/kernels/`](v4/kernels/) contains the learning-oriented implementations:

| File | Contents |
| --- | --- |
| [`recurrent_kda.py`](v4/kernels/recurrent_kda.py) | PyTorch recurrent forward/backward references and Triton recurrent kernels |
| [`chunkwise_kda.py`](v4/kernels/chunkwise_kda.py) | Experimental chunkwise work, including a PyTorch reference |
| [`tests.py`](v4/kernels/tests.py) | Recurrent backward comparisons against PyTorch autograd |

The backward test constructs a real forward state history and checks gradients for Q, K, V, decay gates, update strengths, and the initial state. It tests both zero and nonzero final-state gradients.

Storage and compute dtypes for the custom kernels are configured through `dtypes`, `torch_map`, and `tl_map` in [`v4/config.py`](v4/config.py). This configuration does not automatically control the precision of every FLA or PyTorch operation.

## Getting started

### Environment

Development targets an NVIDIA RTX 3090 with 24 GB VRAM, a Ryzen 5 5600X, and 64 GB DDR4 RAM.

Use a CUDA-enabled PyTorch installation and compatible versions of Triton and flash-linear-attention. A Linux or WSL2 CUDA environment is a practical starting point; consult the upstream FLA installation instructions for supported versions and optional convolution dependencies.

The repository does not yet provide a pinned environment or package installer. V4 also imports Pydantic. After installing CUDA-enabled PyTorch, install the remaining dependencies for your environment, including:

```bash
pip install pydantic flash-linear-attention
```

Follow upstream instructions if your platform needs additional build steps. This command alone is not a complete CUDA environment setup.

```bash
git clone https://github.com/gavmc/Mini-K3.git
cd Mini-K3
```

### Run the recurrent backward test

Run from `v4/` so the current local imports resolve:

```bash
cd v4
python -m kernels.tests
```

The test requires CUDA and Triton. A successful run prints gradient differences and finishes with `All backward gradients match autograd.`

### Run the V3 synthetic benchmark

From the repository root:

```bash
cd v3
python bench.py
```

The benchmark uses random tokens, `torch.compile`, BF16 autocast, and optimizer steps. It reports parameter counts, step time, throughput, and peak allocated GPU memory. Its default sequence length is 16,384; reduce `T` in `bench.py` if needed.

This is a performance probe, not a language-model quality evaluation. FLOP/MFU and token-budget projections are approximate; active-parameter counts are not interchangeable with total parameters or exact compute.

There is no complete V4 training or generation command yet.

## Earlier checkpoint

An earlier model is available at [gav6/mini-k3-125m-sft](https://huggingface.co/gav6/mini-k3-125m-sft).

The model card reports 125.9M total / 63.3M active parameters, 2.39B pretraining tokens, and subsequent instruction tuning. It uses its own generation code rather than the standard Transformers loading interface.

That checkpoint is a research artifact. It can produce fluent text, but instruction following, factual accuracy, repetition, and longer responses remain weak. Improving those behaviors is a central motivation for V4. The checkpoint's results should not be presented as results for the unfinished V4 model.

## Next steps

- [ ] Assemble the V4 backbone, FFNs, and language-model head.
- [ ] Integrate and test model-wide recurrent, convolution, and latent caches.
- [ ] Verify causal behavior and full-sequence versus cached-decoding agreement.
- [ ] Build reproducible data preparation, training, validation, and checkpoint resume paths.
- [ ] Benchmark candidate widths and depths on the RTX 3090.
- [ ] Run short pilots before committing to a full token budget.
- [ ] Compare dense FFNs with MoE at matched total parameters and measured training time.
- [ ] Evaluate instruction following, copying, summarization, repetition, and stopping behavior.

AttnRes, alternative expert activations, and more elaborate routing remain experimental choices for V4 rather than prerequisites for a strong baseline.

## References

- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Applications](https://arxiv.org/abs/2402.14905)
- [SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737)
- [V4 source notes](v4/sources.md)

Built by [Gavin McLaughlan](https://github.com/gavmc).
