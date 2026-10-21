https://blog.doubleword.ai/you-could-have-come-up-with-kimi-delta-attention <- KDA
https://arxiv.org/html/2607.24653 <- Kimi K3 paper
https://arxiv.org/html/2507.10524v3 <- Looped small language models

* use bf16 as opposed to fp16 for better training stability (change if I really need more percision but probably wont be necessary) -> need to look into what other small models do

* need to keep below 255 registers per thread with triton