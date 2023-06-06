'''20230605 check .bin files to see if the file sizes match the header'''
from misc import err, hdr_fn, read_hdr, args
import os

lines = [x.strip() for x in os.popen("ls -1 *.bin").readlines()]

fails = []
for f in lines:
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

    if len(args) > 1:
        for f in fails:
            hf = hdr_fn(f)
            for x in [hf, f]:
                print('rm', x)
                os.remove(x)
else:
    print("All .bin files OK")


