'''bands subselection (every ith band of group of j bands)
python3 raster_bands_mod.py [input binary file name] [ith] [of j]
'''
import math
from misc import read_hdr, run, args, err, hdr_fn, band_names, write_hdr
if len(args) < 4:
    err("python3 raster_bands_mod.py [input binary file] [i e.g. 2] [j e.g. 3] # select every 2nd band of 3")

inf = args[1] # input file
mod = int(args[2]) # ith band to select
mof = int(args[3]) # of j bands

samples, lines, bands = [int(x) for x in read_hdr(hdr_fn(inf))]
bn = band_names(hdr_fn(inf))

if bands % mof != 0:
    err("j must be a divisor of the number of bands")

select = []
select_bn = []
j = mof
for i in range(1, bands + 1):
    x = j * int(math.floor(i / j))
    if i - x == mod:
        select.append(i)
        select_bn.append(bn[i- 1])

c = ' '.join([str(x) for x in select])
print(c)
cmd = 'unstack.exe ' + inf + ' ' + c
run(cmd)
print(select_bn)

out_i = [inf + '_' + str(i).zfill(3) + '.bin' for i in select]
print(out_i)

ofn = inf + '_subselect.bin'
ohn = inf + '_subselect.hdr'
cmd = 'cat ' + ' '.join(out_i) + ' > ' + ofn
print(cmd)
run(cmd)
write_hdr(ohn, str(samples), str(lines), str(len(select)), select_bn)
