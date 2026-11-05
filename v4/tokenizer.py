from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(cfg, train_data, path):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],  # might need to update these at some point
        vocab_size=cfg.vocab_size,
    )

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(train_data, trainer)
    tokenizer.save(path)







    