from datasets import load_dataset, Dataset, Features, Value
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as s3Config
import gzip
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dotenv import load_dotenv
from pathlib import Path

from config import DataConfig



FEATURES = Features(
    {
        "text": Value("string"),
        "source": Value("string"),
        "tokens": Value("int64"),
    }
)


def stack_download_contents(blob_id, s3, bucket_name):
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj['Body']) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise


def _stream(repo, name, cfg: DataConfig):
    return load_dataset(repo, name=name, split="train", streaming=True).shuffle(seed=cfg.seed, buffer_size=cfg.buffer_size)


def fineweb_rows(cfg: DataConfig, budget):
    used = 0
    for ex in _stream("HuggingFaceFW/fineweb-edu", "sample-10BT", cfg):
        n = ex["token_count"] or 0
        yield {"text": ex["text"], "source": "fineweb-edu", "tokens": n}
        used += n
        if used >= budget:
            return


def cosmopedia_rows(cfg: DataConfig, budget):
    used = 0
    for ex in _stream("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", cfg):
        n = ex["token_length"] or 0
        yield {"text": ex["text"], "source": "cosmopedia-v2", "tokens": n}
        used += n
        if used >= budget:
            return


def _fetch(blob_id, s3, bucket_name):
    out = stack_download_contents(blob_id, s3, bucket_name)
    if not out.get("download_success"):
        return None
    return out.get("text") or None


def stack_rows(cfg: DataConfig, budget):
    load_dotenv()
    used = 0
    window = cfg.num_threads * 4
    pending: deque = deque()

    s3 = boto3.client("s3", config=s3Config(max_pool_connections=cfg.num_threads * 2))
    bucket_name = cfg.s3_bucket

    def row_from(ex: dict, fut) -> dict | None:
        text = fut.result()
        if not text:
            return None
        n = max(1, int(len(text) / cfg.bytes_per_token))
        return {"text": text, "source": "stack-edu-Python", "tokens": n}

    pool = ThreadPoolExecutor(max_workers=cfg.num_threads)
    try:
        stream = _stream("HuggingFaceTB/stack-edu", "Python", cfg)
        for ex in stream:
            if ex["length_bytes"] > cfg.max_file_bytes:
                continue
            if ex["int_score"] < cfg.min_int_score:
                continue
            pending.append((ex, pool.submit(_fetch, ex["blob_id"], s3, bucket_name)))
            while len(pending) >= window:
                row = row_from(*pending.popleft())
                if row is None:
                    continue
                used += row["tokens"]
                yield row
                if used >= budget:
                    return
        while pending:
            row = row_from(*pending.popleft())
            if row is None:
                continue
            used += row["tokens"]
            yield row
            if used >= budget:
                return
    finally:
        pool.shutdown(wait=False, cancel_futures=True)



def write_data(name, gen, cfg: DataConfig):
    ds = Dataset.from_generator(gen, features=FEATURES, split="train")
    path = cfg.out_dir / f"{name}.parquet"
    ds.to_parquet(str(path))
    total = sum(ds["tokens"])
    print(f"{name} rows={len(ds)} tokens={total} -> {path}")
    return path


def build(cfg: DataConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("fineweb-edu", lambda: fineweb_rows(cfg, cfg.budget(0))),
        ("cosmopedia-v2", lambda: cosmopedia_rows(cfg, cfg.budget(1))),
        ("stack-edu-Python", lambda: stack_rows(cfg, cfg.budget(2))),
    ]

    for name, gen in jobs:
        write_data(name, gen, cfg)


    

if __name__ == "__main__":
    cfg = DataConfig(total_tokens=100_000_000, buffer_size=100_000, out_dir=Path("data/sample-100M"))
    build(cfg)





    




    




