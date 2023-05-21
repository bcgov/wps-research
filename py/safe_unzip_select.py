'''20230520 unzip sentinel2 files, just for tiles that are "on fire" according to bcws data

20230515 unzip sentinel2 files that are not already unzipped
'''
import os
import sys
from misc import parfor, sep

select_file = '/home/' + os.popen('whoami').read().strip() + sep + 'GitHub' + sep + 'wps-research' + sep + 'py' + sep + '.tiles_select'
select = open(select_file).read().strip().split()

print(select)

cmds = []

for row in select:
	print(row)
	files = [x.strip() for x in os.popen('ls -1 *.zip | grep ' + row).readlines()]
	# only take largest file for this tile for today.

	#by_size = [[os.stat(f).st_size, f] for f in files]
	#by_size.sort(reverse=True)  # decreasing order
	#print(by_size)
	#f = by_size[0][1]
	for f in files:
		d = f[:-4] + '.SAFE'
		if not os.path.exists(d):
			cmds += ['unzip ' + f]

def run(c):
	return os.system(c)

parfor(run, cmds, 4)