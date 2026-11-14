from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
import numpy as np
import json



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
    tokenizer.save(path)


def tokenize_dataset(tok_file, files, path):
    fineweb = load_dataset("parquet", data_files=f"data/sample-100M/{files[0]}.parquet", split="train", streaming=True)
    cosmo = load_dataset("parquet", data_files=f"data/sample-100M/{files[1]}.parquet", split="train", streaming=True)
    stack = load_dataset("parquet", data_files=f"data/sample-100M/{files[2]}.parquet", split="train", streaming=True)


    tokenizer = Tokenizer.from_file(tok_file)
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    meta = {"tokens": [0, 0, 0], "tokenizer": tok_file, "EOS": "<|endoftext|>"}

    for i, source in enumerate([fineweb, cosmo, stack]):
        data_text = batch_iterator(source)
        with open(f"{path}/{files[i]}.bin", "wb") as f:
            for texts in data_text:
                enc = tokenizer.encode_batch(texts, add_special_tokens=False)

                for encoding in enc:
                    ids = encoding.ids
                    ids.append(eos_id)

                    tokens = np.asarray(ids, dtype="<u2")
                    meta["tokens"][i] += len(tokens)
                    tokens.tofile(f)

    with open(f"{path}/metadata.json", "w") as f:
        json.dump(meta, f)

    




if __name__ == "__main__":
    
    tokenize_dataset("tokenizer/tokenizer-32K.json", ["fineweb-edu", "cosmopedia-v2", "stack-edu-Python"], "data/tokenized/sample-100M")

    
    