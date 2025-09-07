import os
import sys
from misc import err

def get(c):
    return os.popen(c).read()

x,y = [int(x) for x in get('xdpyinfo | grep dimensions').strip().split()[1].split('x')]

x2, y2 = None, None
print(get('xrandr | grep primary'))
print(get('xrandr | grep rdp'))
try:
    print(get('xrandr | grep primary'))
    [x2, y2] =[int(x) for x in get('xrandr | grep primary').split()[3].split('+')[0].split('x')]
except:
    pass

try:
    print(get('xrandr | grep rdp0'))
    [x2, y2] =[int(x) for x in get('xrandr | grep rdp').split()[3].split('+')[0].split('x')]
except:
    pass

x_min, y_min = min(x, x2), min(y, y2)
print('.screen_min_x=', x_min)
print('.screen_min_y=', y_min)

open('.screen_min_x', 'wb').write(str(x_min).encode())
open('.screen_min_y', 'wb').write(str(y_min).encode())
