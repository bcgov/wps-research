'''20231024 open tabs in parallel

tabs are opened for each folder name listed in the command arguments
'''
from misc import run
import sys
args = sys.argv

for i in range(1, len(args)):
    x = args[i]

    cmd = 'gnome-terminal --tab -- bash -c "cd ' + x + '; exec bash"'
    run(cmd)
