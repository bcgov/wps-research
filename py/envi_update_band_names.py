# transfer band names from one file to another. Useful if you run a program that throws band name info away!
import os
import sys
from misc import *
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep

if len(args) < 3:
    err('envi_update_band_names.py [.hdr file with band names to use] ' +
        '[.hdr file with band names to overwrite]')

if not exists(args[1]) or not exists(args[2]):
    err('please check input files:\n\t' + args[1] + '\n\t' + args[2])

# need to run this first to make sure the band name fields are where we expect!
run('python3 ' + pd + 'envi_header_cleanup.py ' + args[1])
run('python3 ' + pd + 'envi_header_cleanup.py ' + args[2]) # should really call directly, whatever

i_dat, o_dat = open(args[1]).read(),  open(args[2]).read()

def get_band_names_lines(hdr):
    idx = get_band_names_line_idx(hdr)
    lines = open(hdr).readlines()
    return [lines[i] for i in idx], idx

[bn1, ix1], [bn2, ix2] = get_band_names_lines(args[1]),\
                         get_band_names_lines(args[2])

lines = o_dat.strip().split('\n')

ix = 0
for i in range(0, len(lines)):
    line = lines[i]  # for every line in the output file...
    if i in ix2:  # if it's supposed to be a band-names line!
        lines[i] = bn1[ix].rstrip()  # replace it with the band-names line..
        ix += 1
open(args[2], 'wb').write('\n'.join(lines).encode()) # write the result
