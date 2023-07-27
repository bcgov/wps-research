'''was in active/ need to adjust
'''
import os
import sys
import shutil
import datetime

today = datetime.date.today()
today = str(today.year).zfill(4) + str(today.month).zfill(2) + str(today.day).zfill(2)
d = '/home/' + os.popen('whoami').read().strip() + '/tmp/' + today
if not os.path.exists(d):
    os.mkdir(d)
d += '/'

fire_centre = ['V', 'G', 'C', 'K', 'N', 'R'] 

lines = [] 

for f_c in fire_centre:
	lines += [x.strip() for x in os.popen('ls -1 | grep ' + f_c).readlines()]
print(lines)

watch = set()
for line in lines:
		x = os.popen(' find ./' + line + ' -name "*L2A*zip"').readlines()
		for y in x:
				f = y.split('/')[-1].strip()
				T = f.split('_')[5]
				print(T, f.strip())
				watch.add(T)

print("Watch", watch)



lines = [x.strip() for x in os.popen('find ./ -name "*.zip"').readlines()]
for line in lines:
		f = line.split('/')[-1].strip()
		w = f.split('_')
		if w[0] not in ['S2A', 'S2B']:
				continue
	
		T = w[2]
		tile_id = w[5]
		t = T.split('T')[0]
		if t == today and tile_id in watch:
				print(line)
				df = d + f
				if not os.path.exists(df):
						print("+w", df)
						shutil.copyfile(line, df)	
		
print(today)
