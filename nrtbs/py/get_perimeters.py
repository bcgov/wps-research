import datetime
import urllib.request
import shutil
import zipfile
import ssl
import certifi
import sys

'''
Downloads the current fire perimeters as a zip file
'''
# Define the filename and download path
fn = 'prot_current_fire_polys.zip'
dl_path = 'https://pub.data.gov.bc.ca/datasets/cdfc2d7b-c046-4bf0-90ac-4897232619e1/' + fn

# Create an SSL context using certifi
context = ssl.create_default_context(cafile=certifi.where())

# Download the file using urllib with SSL context
with urllib.request.urlopen(dl_path, context=context) as response, open(fn, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

# Create a backup of the downloaded file with a timestamp
# shutil.copyfile(fn, 'prot_current_fire_polys' + '.zip')

# Extract the contents of the zip file
with zipfile.ZipFile('prot_current_fire_polys' + '.zip', 'r') as zip_ref:
    zip_ref.extractall('.') 

fn_2 = 'prot_current_fire_points.zip'
dl_path2 = 'https://pub.data.gov.bc.ca/datasets/2790e3f7-6395-4230-8545-04efb5a18800/' + fn_2

# Create an SSL context using certifi
context = ssl.create_default_context(cafile=certifi.where())

with urllib.request.urlopen(dl_path2, context=context) as response, open(fn_2, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

# shutil.copyfile(fn_2, 'prot_current_fire_points' + '.zip')

with zipfile.ZipFile('prot_current_fire_points' + '.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

with urllib.request.urlopen(dl_path2, context=context) as response, open(fn_2, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)


print("Download and extraction complete.")
