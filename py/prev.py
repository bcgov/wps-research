'''20220910 prev.py: find previous result, if available'''
import os
from misc import run, err, runlines, sep
pd = os.path.abspath(runlines('pwd')[-1])  # get present directory

lines = runlines("ls -latrh .. | grep '^d'")
lines = [x.split() for x in lines]
last = None  # most recent folder other than this one
for line in lines:
    a = line[-1]
    if a != '.' and a != '..':
        x = os.path.abspath('..' + sep + a)
        if x != pd:
            last = x
            # print(x)

if last:
    print("Earlier result folder:", last)
    lines = runlines("ls -latrh " + last + sep + "*.bin")
    last_f = None
    for line in lines:
        w = line.split()[-1]
        wf = w.split(sep)[-1] # print(wf, wf[:3])
        if wf[:3] != 'sub':
            last_f = w
print("guess:", last_f)
