import os
import sys
from misc import err

def get(c):
    return os.popen(c).read()

x,y = [int(x) for x in get('xdpyinfo | grep dimensions').strip().split()[1].split('x')]

[x2, y2] =[int(x) for x in get('xrandr | grep primary').split()[3].split('+')[0].split('x')]
x_min, y_min = min(x, x2), min(y, y2)

open('.screen_min_x', 'wb').write(str(x_min).encode())
open('.screen_min_y', 'wb').write(str(y_min).encode())
