'''
  query sentinel-2 products over a given:
    1) point-- lat, long
    2) a TILE ID of interest
'''

import os
import sys

user_, pass_ = None, None
if not os.path.exists('./.user'):
    user_ = input('please enter your copernicus username:').strip()
    open('./.user', 'wb').write(user_.encode())
else:
    user_ = open('./.user', 'rb').read().decode()

if not os.path.exists('./.pass'):
    pass_ = input('please enter your copernicus password:').strip()
    open('./.pass', 'wb').write(pass_.encode())
else:
    pass_ = open('./.pass', 'rb').read().decode()


cmd = ('wget --no-check-certificate --user=' + user_ + ' --password=' + pass_ +
        ' --output-document=out.html "https://scihub.copernicus.eu/dhus/search?q=(platformname:Sentinel-2 AND footprint:\\"Intersects(51.0602686,-120.9083258)\\")"')

print(cmd)
print(os.popen(cmd).read())
