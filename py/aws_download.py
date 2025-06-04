'''20250604 test downloading from AWS S3 using python api ( boto 3) 
'''

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from tqdm import tqdm
import os
import time

# Settings
BUCKET = "sentinel-products-ca-mirror"
KEY = "Sentinel-2/S2MSI2A/2025/05/27/S2C_MSIL2A_20250527T191931_N0511_R099_T10VFL_20250528T002013.zip"
LOCAL_PATH = "L2_T10VFL/" + os.path.basename(KEY)
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 500

# Ensure output directory exists
os.makedirs(os.path.dirname(LOCAL_PATH), exist_ok=True)

# Configure anonymous client with retry settings
config = Config(
    signature_version=UNSIGNED,
    retries={'max_attempts': MAX_RETRIES, 'mode': 'standard'}
)
s3 = boto3.client("s3", config=config)

# Get total file size
try:
    head = s3.head_object(Bucket=BUCKET, Key=KEY)
    total_size = head['ContentLength']
except Exception as e:
    print("Error getting object metadata:", e)
    raise

# Resume support
existing_size = 0
if os.path.exists(LOCAL_PATH):
    existing_size = os.path.getsize(LOCAL_PATH)

if existing_size >= total_size:
    print("File already fully downloaded.")
    exit(0)

# Download loop with progress
with open(LOCAL_PATH, "ab") as f, tqdm(
    total=total_size, initial=existing_size, unit="B", unit_scale=True, desc="Downloading"
) as pbar:
    start = existing_size
    while start < total_size:
        end = min(start + CHUNK_SIZE - 1, total_size - 1)
        byte_range = f"bytes={start}-{end}"

        for attempt in range(MAX_RETRIES):
            try:
                resp = s3.get_object(Bucket=BUCKET, Key=KEY, Range=byte_range)
                data = resp['Body'].read()
                f.write(data)
                f.flush()
                pbar.update(len(data))
                break
            except (BotoCoreError, ClientError) as e:
                print(f"Retry {attempt + 1} failed for bytes {start}-{end}: {e}")
                time.sleep(1)
        else:
            raise RuntimeError(f"Failed to download range {byte_range} after {MAX_RETRIES} retries")

        start += CHUNK_SIZE

print("Download complete.")

