'''20230605 check .bin files to see if the file sizes match the header'''
from misc import err, hdr_fn, read_hdr, args, exist, args
import os

lines = None
if '-r' in args or '--recursive' in args:
    lines = [x.strip() for x in os.popen('find ./ -name "*.bin"').readlines()]
else:
    # non-recursive ( default ) 
    lines = [x.strip() for x in os.popen("ls -1 *.bin").readlines()]

fails = []
no_hdr = []
for f in lines:
    hfn = f[:-4] + '.hdr'
    print(hfn)
    try:
        hfn = hdr_fn(f)
    except:
        no_hdr += [f]

    if not exist(hfn):
        print('[ERR]', hfn, 'not found')
    else:
        [samples, lines, bands] = [int(x)
                                   for x in read_hdr(hdr_fn(f))]
    f_size = os.stat(f).st_size
    expected = samples * lines * bands * 4 

    if f_size == expected:
        print(f, '[OK]')
    else:
        print(f, '[BAD]', f_size, '/', expected)
        fails += [f]

if len(fails) > 0:
    print("Checksum failed for files:")
    print(' '.join(fails))

    if True: # len(args) > 1:
        for f in fails:
            print(f)
            hfn = f[:-4] + '.hdr'
            try:
                hfn = hdr_fn(f)
            except:
                pass
            for x in [hfn, f]:
                if exist(x):
                    print('rm', x)
                    os.remove(x)

if len(no_hdr) > 0:
    print("These files had no header:")
    print(' '.join(no_hdr))
    if len(args) > 1:
        for f in no_hdr:
            print('rm', f)
            os.remove(f)
else:
    print("All .bin files had header")


print("bad files:")
print(' '.join(fails))

