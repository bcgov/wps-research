from misc import run, err, exists, hdr_fn, args, band_names
import os

if len(args) < 2:
    err("ts_stack_change.py [stack.bin produced from SNAP stack operator, followed by snap2psp.py ./ 1")

fn = args[1]
hfn = hdr_fn(fn)
bn = band_names(hfn)

phone_book = {}
lookup = {bn[i]: i for i in range(len(bn))}

for b in bn:
    w = b.split('_')
    ti = w[0].split('T')[1]
    
    if ti[0] == '4' or ti[1] == '4':
        pass  # would be: "continue" instead of "pass" to use T3 elements only
    
    if not w[-1] in phone_book:
        phone_book[w[-1]] = []
    phone_book[w[-1]] += [b]

for p in phone_book:
    if not exists(p):
        os.mkdir(p)
    for b in phone_book[p]:
        w = b.split("_")
        ofn = w[-1] + '/' + ('_'.join(w[:-2])) + '.bin'
        ohn = w[-1] + '/' + ('_'.join(w[:-2])) + '.hdr'
        cmd = ' '.join(['unstack2',
                        fn,
                        str(lookup[b]),
                        ofn])
        run(cmd)

        run('raster_zero_to_nan ' + ofn)
