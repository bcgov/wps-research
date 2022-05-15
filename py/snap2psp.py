'''20220515 CONVERT FROM ESA SNAP, to raw binary format. i.e.,:

Convert .img files (with byte-order =1) to:
        .bin files (with byte order 0)
by swapping byte order..
..this is for converting .img files produced by SNAP, to PolSARPro format files

20220515: update this to read/write headers by copy/modify

NOTE: can run this on the .data portion of a "dimap" format dataset in SNAP'''
import os
import sys
from misc import args, sep, run, parfor, exist, pd

create_stack = False
d = os.getcwd()  # default: run in present directory
if len(args) < 2:
    print('snap2psp.py [input folder name] # [optional arg: stack thebands] ' +
          '# convert snap byte-order= 1 .img data to byte-order 0 .bin data')
else:
    d = args[1]
    create_stack = True

cmds, hdrs, out_files = [], [], []
p = os.path.abspath(d) + sep
cmd = 'ls -1 ' + p + '*.img'
files = [x.strip() for x in os.popen(cmd).readlines()]
for f in files:
    if os.path.isfile(f):
        of = f[:-4] + '_envi.bin'
        hf0, hf1 = f[:-3] + 'hdr', of[:-3] + 'hdr'
        if not exist(of):
            cmds.append('sbo ' + f + ' ' + of + ' 4')
        if not exist(hf1):
            cmds.append('cp -v ' + hf0 + ' ' + hf1)
        out_files.append(of)
        hdrs.append(hf1)
    else:
        err('not file:' + f)

parfor(run, cmds, 4)

# update the new headers to reflect the change in byte order
for h in hdrs:
    print('+w', h)
    dat = open(h).read().replace('byte order = 1',
                               'byte order = 0')
    open(h, 'wb').write(dat.encode())

# stack the files into a BSQ stack?
stack_f = p + 'stack.bin'
if create_stack and not exist(stack_f):
    run(['python3 ' + pd + 'raster_stack.py'] +
         out_files +
         [stack_f])
