https://blog.doubleword.ai/you-could-have-come-up-with-kimi-delta-attention -> 4.1

* use bf16 as opposed to fp16 for better training stability (change if I really need more percision but probably wont be necessary) -> need to look into what other small models do

* need to keep below 255 registers per thread with triton