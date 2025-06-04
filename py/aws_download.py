'''20250604 test downloading from AWS S3 using python api ( boto 3) 
'''
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import logging
import os


def aws_download(BUCKET, KEY, LOCAL_PATH):
    # === CONFIGURATION ===
    # BUCKET = "sentinel-products-ca-mirror"
    # KEY = "Sentinel-2/S2MSI2A/2025/05/27/S2C_MSIL2A_20250527T191931_N0511_R099_T10VFL_20250528T002013.zip"
    # LOCAL_PATH = Path("L2_T10VFL") / Path(KEY).name
    CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
    MAX_RETRIES = 10
    NUM_THREADS = 8  # <== You can change this value

    # === SETUP ===
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = Config(signature_version=UNSIGNED, retries={'max_attempts': MAX_RETRIES, 'mode': 'standard'})
    s3 = boto3.client("s3", config=config)

    # === GET FILE SIZE ===
    try:
        total_size = s3.head_object(Bucket=BUCKET, Key=KEY)['ContentLength']
    except Exception as e:
        logging.error("Error getting object metadata: %s", e)
        raise SystemExit(1)

    # === DEFINE CHUNKS ===
    def get_chunks(start, end, size):
        return [(i, min(i + size - 1, end)) for i in range(start, end + 1, size)]

    # === DOWNLOAD FUNCTION ===
    def download_chunk(start, end):
        byte_range = f"bytes={start}-{end}"
        for attempt in range(MAX_RETRIES):
            try:
                resp = s3.get_object(Bucket=BUCKET, Key=KEY, Range=byte_range)
                data = resp['Body'].read()
                return start, data
            except (BotoCoreError, ClientError) as e:
                logging.warning(f"Retry {attempt + 1} failed for range {byte_range}: {e}")
        raise RuntimeError(f"Failed to download range {byte_range} after {MAX_RETRIES} retries")

    # === RESUME SUPPORT ===
    existing_size = LOCAL_PATH.stat().st_size if LOCAL_PATH.exists() else 0
    if existing_size >= total_size:
        logging.info("File already fully downloaded.")
        raise SystemExit(0)

    # === MULTITHREADED DOWNLOAD ===
    chunks = get_chunks(existing_size, total_size - 1, CHUNK_SIZE)

    try:
        with open(LOCAL_PATH, "r+b" if LOCAL_PATH.exists() else "wb") as f:
            f.truncate(total_size)  # Pre-allocate file size
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = {executor.submit(download_chunk, start, end): (start, end) for start, end in chunks}
                with tqdm(total=total_size, initial=existing_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
                    for future in as_completed(futures):
                        start, data = future.result()
                        f.seek(start)
                        f.write(data)
                        pbar.update(len(data))
    except KeyboardInterrupt:
        logging.warning("Download interrupted by user.")
        raise SystemExit(1)

    logging.info("Download complete.")


BUCKET = "sentinel-products-ca-mirror"
KEY = 'Sentinel-2/S2MSI2A/2025/06/02/S2C_MSIL2A_20250602T193921_N0511_R042_T10VDM_20250602T231915.zip' # "Sentinel-2/S2MSI2A/2025/05/27/S2C_MSIL2A_20250527T191931_N0511_R099_T10VFL_20250528T002013.zip"
LOCAL_PATH = Path("L2_T10VDM") / Path(KEY).name

aws_download(BUCKET, KEY, LOCAL_PATH)

