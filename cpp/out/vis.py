import os
import sys

def err(m):
    print("Error:", m)
    sys.exit(1)

args = sys.argv
f = args[1]

try:
    i = int(f)
except Exception:
    err("please enter a number: classification iteration number")

f = str(int(f))
fn = f + '.bin'
if not os.path.exists(fn):
    fn = f + '.out'
    if not os.path.exists(fn):
        err("file does not exist:" + fn)

def run(c):
    print(c)
    a = os.system(c)
    if(a != 0): err("cmd failed" + c)
run('cp ' + fn + " " + fn[:-4] + '.bin')

run("grep -n n_class ../n_class.csv")
run("grep -n " + f + " ../n_class.csv")

run('cp hdr ' + f + '.hdr')
# run('cp ' + fn + ' ' + f + '.bin')
# run('../class_wheel.exe ' + f + '.bin')
run('abn ' + f + '.hdr 5')
run('imv ' + f + '.bin')
