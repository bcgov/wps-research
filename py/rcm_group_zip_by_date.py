'''20220515 group RCM data (zip) by date 
* Separate folders'''
from misc import * 

zips = os.popen('ls -1 RCM*.zip').readlines()
dates = list(set([z.strip().split('_')[5] for z in zips]))

for d in dates:
    if not exist(d):
        os.mkdir(d)

for z in zips:
    z = z.strip()
    w = z.split('_')
    d = w[5]
    run(['mv -v',
         z,
         d])
