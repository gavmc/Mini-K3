from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset



def batch_iterator(ds, batch_size=1000):
    for batch in ds.iter(batch_size=batch_size):
        yield batch["text"]

def train_tokenizer(vocab_size, train_data, path):
    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = BpeTrainer(
        special_tokens=["<|im_start|>", "<|im_end|>", "<|pad|>", "<|endoftext|>", "<think>", "</think>"],
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    tokenizer.train_from_iterator(train_data, trainer)
    #tokenizer.train(train_data, trainer)
    tokenizer.save(path)





if __name__ == "__main__":

    ds = load_dataset("data/sample-100M", split="train", streaming=True)
    train_tokenizer(32000, batch_iterator(ds), "tokenizer/tokenizer-32K.json")

    
    