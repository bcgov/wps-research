'''raster_band_select.py:
band-subselection by matching on a string (such as by date)
'''
from misc import args, sep, exists, pd, run, hdr_fn, band_names, err, read_hdr

if len(args) < 3:
    err('raster_band_select.py [input binary file] ' +
        '[string to match band names e.g. 20210721 for jul 21 2021] [additional string to match on..]')

fn, s = args[1], args[2:]
cd = pd + '..' + sep + 'cpp' + sep

hfn = hdr_fn(fn)
samples, lines, bands = read_hdr(hfn)
nrow, ncol = lines, samples

bi = []
bn = band_names(hfn)
for i in range(len(bn)):
    match = False
    for pattern in s:
        if len(bn[i].split(pattern)) > 1:
            match = True
    if match:
        bi.append(i)
        print('  ', str(i + 1), '  ', bn[i])

if not exists(fn + '_001.hdr'):
    run(cd + 'unstack.exe ' + fn)

for i in bi:
    f = fn + '_' + str(i + 1).zfill(3) + '.bin'
    print(f)
fni = [fn + '_' + str(i + 1).zfill(3) + '.bin' for i in bi]
bands = str(len(bi))  # number of bands to write
ofn = fn + '_band_select.bin'
ohn = fn + '_band_select.hdr'
run('cp ' + hfn + ' ' + ohn)

c = ['python3',
     pd + 'envi_header_modify.py',
     ohn,
     nrow,
     ncol,
     bands]
c += [('"' + bn[bi[i]] + '"') for i in range(len(bi))]
c = ' '.join(c)
run(c)

c = ['cat']
c += fni
c += ['>', ofn]
c = ' '.join(c)
run(c)
