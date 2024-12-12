'''20241111: delete intermediary files from 
alos_dualpol_process.py 

leave data from the last step!!'''

import os

for x in range(1, 7, 1):  # look at processing steps 1 through 6
    xp = str(x).zfill(2)
    cmd = 'find ./ -name "' + xp + '*.dim" | xargs rm -rf'
    print(cmd)
    a = os.system(cmd)


    cmd = 'find ./ -name "' + xp + '*.data" | xargs rm -rf'
    print(cmd)
    a = os.system(cmd)

