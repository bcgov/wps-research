'''20250602 refresh point and perimeter files data from bcws source on data.gov.bc.ca

backup the current copies with timestamp of retrieval'''
from misc import download_file 

# Files to download, back up, and extract
files_to_process = [{'filename': 'prot_current_fire_polys.zip',  # current fires polygon database
                     'url_base': 'https://pub.data.gov.bc.ca/datasets/cdfc2d7b-c046-4bf0-90ac-4897232619e1/',},
                    {'filename': 'prot_current_fire_points.zip',  # current fires points database
                     'url_base': 'https://pub.data.gov.bc.ca/datasets/2790e3f7-6395-4230-8545-04efb5a18800/',}]

# Process each file
for file in files_to_process:
    full_url = file['url_base'] + file['filename']
    download_file(full_url, file['filename'], True)
