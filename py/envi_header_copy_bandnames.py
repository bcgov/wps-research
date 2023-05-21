'''copy map info from one ENVI header file to another'''
import os
import sys
from misc import *
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep

if len(args) < 3:
    err('envi_header_copy_bandnames.py [source .hdr file] [dest .hdr file to modify]')
if not exists(args[1]) or not exists(args[2]): 
    err('please check input files:\n\t' + args[1] + '\n\t' + args[2])

# need to run this first to make sure the band name fields are where we expect!
run('python3 ' + pd + 'envi_header_cleanup.py ' + args[1])
run('python3 ' + pd + 'envi_header_cleanup.py ' + args[2])


samples, lines, bands = read_hdr(args[2])
b_n = band_names(args[1]) #  = [x.strip() for x in os.popen("envi_header_band_names.py " + args[1]).readlines()][1:]


print("b_n", b_n)
c = ' '.join(['envi_header_modify.py',
              args[2],
              str(lines),
              str(samples),
              str(bands)] + ['"' + b.replace(' ', '\\ ') + '"' for b in b_n]) # and_names])
print(c)
sys.exit(1)
run(c)
# err('envi_header_modify.py [.hdr file to modify] [nrow] [ncol] [nband] [band 1 name]... [band n name]')
