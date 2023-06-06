'''20230605 check .bin files to see if the file sizes match the header'''
from misc import err

lines = [x.strip() for x in os.popen("ls -1 *.bin").readlines()]

for line in lines:
    print(line)





