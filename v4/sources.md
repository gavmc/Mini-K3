# Mini-K3 sources

Reading list and implementation references for V4. A listed paper is background reading, not a claim that its approach is implemented or validated in Mini-K3. Current behavior and measurements are summarized in the [README](../README.md).

## Architecture

- [Kimi K3 paper](https://arxiv.org/html/2607.24653) - architectural inspiration.
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) - KDA background.
- [You could have come up with Kimi Delta Attention](https://blog.doubleword.ai/you-could-have-come-up-with-kimi-delta-attention) - explanatory blog post; a secondary source alongside the paper and implementation.
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) - FLA implementation used by the V4 training path, including KDA and short convolutions.
- [Attention Residuals](https://arxiv.org/abs/2603.15031) and [implementation](https://github.com/MoonshotAI/Attention-Residuals) - later experiment; not implemented in V4.
- [Looped small language models](https://arxiv.org/html/2507.10524v3) - further reading retained from the original list.

## Small models and training data research

- [MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Applications](https://arxiv.org/abs/2402.14905) - small-model architecture research.
- [SmolLM2: When Smol Goes Big - Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737) - small-model data and training research.

## Optimization and learning-rate schedules

- [Muon paper](https://arxiv.org/html/2502.16982v1) - optimizer background.
- [PyTorch Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) - API used by `v4/optimization/optim.py`; consult the version matching the installed PyTorch for defaults and LR scaling.
- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) - companion optimizer API.
- [TorchTitan](https://github.com/pytorch/torchtitan) - training implementation reference.
- [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/abs/2404.06395) - WSD schedule reference.
- [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392) - constant-LR and cooldown research.

V4 currently implements linear warmup, a constant phase, and `1 - sqrt(progress)` cooldown. Learning rates and main-run phase lengths remain provisional; these references do not establish a tuned recipe for this model.

## Pretraining datasets

| Dataset card | Configuration used by the downloader | Intended share of training tokens |
| --- | --- | ---: |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `sample-10BT` | 70% |
| [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | `cosmopedia-v2` | 15% |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | `Python` | 15% |

These are Gavin's target proportions by tokens. Download proxy counts differ from the project's tokenizer counts. The current sample contains 126,530,487 actual tokens including EOT, with shares of 72.129%, 13.643%, and 14.228%; source sampling to enforce the intended mix has not been implemented.

## Tokenization and binary storage

- [Hugging Face Tokenizers API](https://huggingface.co/docs/tokenizers/api/tokenizer) - tokenizer loading, batch encoding, and decoding.
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) - parquet loading, streaming, and batch iteration.
- [NumPy binary I/O](https://numpy.org/doc/stable/user/how-to-io.html) - raw array writing and reading.
- [NumPy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) - accessing binary arrays without loading the complete file into RAM.

The project's tokenization code is in [`tokenizer/tokenizer.py`](tokenizer/tokenizer.py). It stores one raw binary stream per source as little-endian uint16 (`<u2`) and appends `<|endoftext|>` after each document. File contents and counts were verified against all source documents on 2026-11-13. Packing and validation splitting remain unfinished.
