from datasets import load_dataset
import boto3
from botocore.exceptions import ClientError
import gzip




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

def tokenizer_sample(size):

    fineweb = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="CC-MAIN-2025-26", 
        split="train", 
        streaming=True
    
    )

    cosmo = load_dataset(
        "HuggingFaceTB/smollm-corpus", 
        name="cosmopedia-v2", 
        split="train", 
        streaming=True
    )

    stack = load_dataset(
        "HuggingFaceTB/stack-edu", 
        name="python", 
        split="train", 
        streaming=True
    )


    




