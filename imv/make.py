import os

def run(cmd):
    print(cmd)
    return os.system(cmd)

run("rm -f imv.exe")
run("make")
which = os.popen("which imv").read().strip()
print("which" + str([which]))
run("cp imv.exe " + which)
