'''20250604 test downloading from AWS S3 using python api ( boto 3) 
'''

import boto3
from botocore.config import Config
from botocore import UNSIGNED
from botocore.client import Config as BotocoreConfig
import os

# S3 URL components
bucket_name = "sentinel-products-ca-mirror"
object_key = "Sentinel-2/S2MSI2A/2025/05/27/S2C_MSIL2A_20250527T191931_N0511_R099_T10VFL_20250528T002013.zip"
local_path = "L2_T10VFL/S2C_MSIL2A_20250527T191931_N0511_R099_T10VFL_20250528T002013.zip"

# Ensure destination folder exists
os.makedirs(os.path.dirname(local_path), exist_ok=True)

# Anonymous S3 client with high retry settings
config = Config(
    retries={
        'max_attempts': 500,
        'mode': 'standard'
    }
)

s3 = boto3.client(
    's3',
    config=config,
    botocore_config=BotocoreConfig(signature_version=UNSIGNED)
)

# Download the file
print(f"Downloading s3://{bucket_name}/{object_key} to {local_path} ...")
s3.download_file(bucket_name, object_key, local_path)
print("Download completed.")
