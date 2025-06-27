import os
from misc import args, run

lines = os.popen("ps | grep " + args[1]).readlines()

lines = [x.strip() for x in lines]

for line in lines:
    w = line.split()[0]

    cmd = "sudo kill " + w
    print(cmd)
    run(cmd)
