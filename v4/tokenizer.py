from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def train_tokenizer(cfg, train_data, path):
    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = BpeTrainer(
        special_tokens=["<|im_start|>", "<|im_end|>", "<|pad|>", "<|endoftext|>", "<think>", "</think>"],
        vocab_size=cfg.vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    tokenizer.train(train_data, trainer)
    tokenizer.save(path)







    