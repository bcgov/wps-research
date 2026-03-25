'''20260302 force restart remote desktop
'''
import os

def run(c):
    print(c)
    return os.system(c)

cmds = ['sudo systemctl stop xrdp',
        'sudo systemctl stop xrdp-sesman',
        'sudo killall Xorg',
        'sudo systemctl start xrdp',
        'sudo systemctl start xrdp-sesman']

for c in cmds:
    a = run(c)
