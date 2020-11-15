# this script should delete all the object files and exe files, so that everything recompiles from scratch (in parallel)
import os

def run(cmd):
    print(cmd)
    return os.system(cmd)

run("clean")
run("rm -f imv.exe")
run("./make.sh")
run("chmod 755 imv.exe")
which = os.popen("which imv").read().strip()
print("which" + str([which]))
run("cp imv.exe " + which)
