import os

lines = os.popen("ls -1 S2*.png").readlines()
lines = [x.strip() for x in lines]

for line in lines:
    T = line.split('_')[2].split('T')[0]
    cmd = 'mv -v ' + line + ' ' + T + '_' + line
    print(cmd)
    a = os.system(cmd)
