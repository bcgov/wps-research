'''20250602 refresh point and perimeter files data from bcws source on data.gov.bc.ca

backup the current copies with timestamp of retrieval'''
import urllib.request
import shutil
import zipfile
import ssl
import certifi
import datetime
import os

# Files to download, back up, and extract
files_to_process = [{'filename': 'prot_current_fire_polys.zip',  # current fires polygon database
                     'url_base': 'https://pub.data.gov.bc.ca/datasets/cdfc2d7b-c046-4bf0-90ac-4897232619e1/',},
                    {'filename': 'prot_current_fire_points.zip',  # current fires points database
                     'url_base': 'https://pub.data.gov.bc.ca/datasets/2790e3f7-6395-4230-8545-04efb5a18800/',}]

def download_file(url: str, filename: str, context, timestamp: str):
    """Download a file and save a timestamped backup."""
    with urllib.request.urlopen(url, context=context) as response, open(filename, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"+w {filename}")
    # Save a timestamped backup
    base, ext = os.path.splitext(filename)
    backup_filename = f"{base}_{timestamp}{ext}"
    shutil.copyfile(filename, backup_filename)
    print(f"+w {backup_filename}")

def extract_zip(filename: str, extract_to: str = '.'):
    """Extract a zip file to the given directory."""
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted: {filename}")

# Timestamp for backups
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")

# SSL context using certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Process each file
for file in files_to_process:
    full_url = file['url_base'] + file['filename']
    download_file(full_url, file['filename'], ssl_context, timestamp)
    extract_zip(file['filename'])

