'''
  query sentinel-2 products over a given:
    1) point-- lat, long
    2) a TILE ID of interest?
'''

import os
import sys

# save username and password to files:
user_, pass_ = None, None
if not os.path.exists('./.user'):
    user_ = input('please enter your copernicus username:').strip()
    open('./.user', 'wb').write(user_.encode())
    print('username written to ./.user')
else:
    user_ = open('./.user', 'rb').read().decode()

if not os.path.exists('./.pass'):
    pass_ = input('please enter your copernicus password:').strip()
    open('./.pass', 'wb').write(pass_.encode())
    print('password written to ./.pass')
else:
    pass_ = open('./.pass', 'rb').read().decode()

def c(add= ''):
    cmd = ('wget --no-check-certificate --user=' + user_ + ' --password=' + pass_ +
           ' --output-document=out.html "https://scihub.copernicus.eu/dhus/search?q=(platformname:Sentinel-2 AND footprint:\\"Intersects(51.0602686,-120.9083258)\\")' + add + '"')
    return cmd


cmd = c()
print(cmd)
r = os.popen(cmd).read()

# read in result
dat = open('out.html').read()
lines = dat.strip().split('\n')

n_results = -1
# search for number of results: <opensearch:totalResults>563</opensearch:totalResults>
for i in range(0, len(lines)):
    line = lines[i].strip()
    if line[0:25] == '<opensearch:totalResults>':
        # sanity check
        if line[-26:] != '</opensearch:totalResults>':
            err('expected end tag')
        w = line.split('<opensearch:totalResults>')[1]
        w = w.split('</opensearch:totalResults>')[0]
        n_results = int(w)


            

