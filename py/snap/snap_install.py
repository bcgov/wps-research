'''install snap version 8...

link to esa-snap, not snap (solve ubuntu issue)

'''
from misc import run, exists, pd, sep
use_mirror = False
main = 'https://download.esa.int/step/snap/8.0/installers/esa-snap_all_unix_8_0.sh'
mirror = 'https://step.esa.int/downloads/8.0/installers/esa-snap_all_unix_8_0.sh'

path = mirror if use_mirror else main
fn = path.split('/')[-1]
target = pd + 'snap' + sep + fn
print(target)
