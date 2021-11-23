#!/usr/bin/python3
import os

def run(cmd):
    print(cmd)
    return os.system(cmd)

run("sudo touch /usr/bin/imv_lite") # where the command will go
run("sudo chmod 755 /usr/bin/imv_lite") # make it runnable

run("clean")
run("rm -f imv_lite.exe")
run("./make.sh")
run("chmod 755 imv_lite.exe")
which = os.popen("which imv_lite").read().strip()
print("which" + str([which]))
run("sudo cp imv_lite.exe " + which)
