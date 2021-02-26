#!/usr/bin/python3
import os

def run(cmd):
    print(cmd)
    return os.system(cmd)

run("sudo touch /usr/bin/imv") # where the command will go
run("sudo chmod 755 /usr/bin/imv") # make it runnable

run("clean")
run("rm -f imv.exe")
run("./make.sh")
run("chmod 755 imv.exe")
which = os.popen("which imv").read().strip()
print("which" + str([which]))
run("sudo cp imv.exe " + which)
