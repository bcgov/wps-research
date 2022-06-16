'''update an ENVI header file:
    - band names portion
    - image dimensions or number of bands
(updated 20220324)'''
import os
import sys
from misc import *
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep

if len(args) < 6:
    err('envi_header_modify.py [.hdr file to modify] [nrow] [ncol] [nband] [band 1 name]... [band n name]')

nrow, ncol, nband = args[2], args[3], args[4]
if not exists(args[1]): 
    err('please check input files:\n\t' + args[1] + '\n\t' + args[2])

# need to run this first to make sure the band name fields are where we expect!
if len(args) < int(nband) + 5:
    run('python3 ' + pd + 'envi_header_cleanup.py ' + args[1])
lines = open(args[1]).read().strip().split('\n')

def get_band_names_lines(hdr):
    idx = get_band_names_line_idx(hdr)
    lines = open(hdr).readlines()
    return [lines[i] for i in idx], idx
[bn1, ix1] = get_band_names_lines(args[1])

lines_new = []
for i in range(0, len(lines)):
    line = lines[i]  # for every line in the output file...
    
    w = [x.strip() for x in line.split('=')]
    if len(w) > 1:
        if w[0] == 'samples': line = 'samples = ' + ncol
        if w[0] == 'lines': line = 'lines = ' + nrow
        if w[0] == 'bands': line = 'bands = ' + nband

    if i not in ix1:  # if it's a band-names line!
        lines_new.append(line)

# write new header file
bn_new = args[5: 5 + int(nband)]
if len(bn_new) != int(nband):
    err('inconsistent input')

lines_new += ['band names = {' + bn_new[0]]
print([bn_new[0]])
for i in range(1, len(bn_new)):
    lines_new[-1] += ','
    print([bn_new[i]])
    lines_new += [bn_new[i]]
lines_new[-1] += '}'
print('+w', args[1])
# print(lines_new); sys.exit(1)
open(args[1], 'wb').write('\n'.join(lines_new).encode())
