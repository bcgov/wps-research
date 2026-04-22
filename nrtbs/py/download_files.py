'''
takes a string tile ID eg. 'T10UFB', and a string date range of integer form yyyymmdd and downloads and extracts the SWIR NIR bands of the avaialbe sentinel 2 data. Only takes frames with cloud cover less then 10 %
>>> download_urls_from_tileid('T10UFB', 20240701, 20240730)
'''
import os
from sentinel2_extract_swir_nir import extract


def download_urls_from_tileid(tile_id, start_date, end_date):
    '''
    takes a string tile ID eg. 'T10UFB', and a string date range of integer form yyyymmdd
    '''
    ID = []
    URL = []
    x = open('index.csv:PRODUCT_ID.txt', 'r').read().split('\n')
    y = open('index.csv:BASE_URL.txt', 'r').read().split('\n')
    z = open('index.csv:CLOUD_COVER', 'r').read().split('\n')
    # searching for valid frames
    matches = []
    for i in range(len(x)-1):
        date = x[i+1].split('_')[2].split('T')[0]
        if date == 'PRD':
            continue
        if x[i+1].split('_')[5] == tile_id and start_date <= int(date) <= end_date and float(z[i+1]) <= 10:
            print(z[i+1])
            matches += [i+1]
    if len(matches) == 0:
        print('No suitible frames in date range')
        
    #extracting data for valid mathces
    for i in matches:
        ID.append(x[i])
        URL.append(y[i])
        if not os.path.exists(x[i] + '.SAFE'):
            os.mkdir(x[i] + '.SAFE')
        cmd = ' '.join(['gsutil -m',
                            'rsync -r',
                            y[i].strip(),
                            x[i] + '.SAFE']) 
        a = os.system(cmd)
        
        file_name  = x[i] + ".SAFE"
        extract(file_name)
    return [ID,URL]


