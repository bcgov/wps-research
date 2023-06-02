'''20230515 unzip sentinel2 files that are not already unzipped
'''
import os
import sys
from misc import parfor

cmds = []
files = [x.strip() for x in os.popen('ls -1 *.zip').readlines()]
for f in files:
	d = f[:-4] + '.SAFE'
	if os.path.exists(d):
		pass
	else:
		cmds += ['unzip ' + f]

def run(c):
	return os.system(c)

parfor(run, cmds, 8)
