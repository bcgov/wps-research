'''
20240712 subset image with GDAL, cleanup ENVI header
Called on a file or file directory.
will trim all files in directory
'''
from misc import run, err, args
from envi import envi_header_copy_bandnames
import os

def cut(fn, A, B, C, D):
    fire_num = fn.split('/')[0]

    if not os.path.exists(f'{fire_num}_cut'):
        os.mkdir(f'{fire_num}_cut')

    out_fn = f'{fire_num}_cut/{fn.split("/")[1].strip(".bin")}_cut.bin'
    out_fn_hdr = f'{fire_num}_cut/{fn.split("/")[1].strip(".bin")}_cut.hdr'

    if os.path.exists(out_fn) and os.path.exists(out_fn_hdr):
        print("skip cut", fn, A, B, C, D)
        return

    run(f'gdal_translate -of ENVI -ot Float32 -srcwin { (" ".join([A, B, C, D]))} {fn} {out_fn}')  # cuting file

    envi_header_copy_bandnames(['',fn[:-4] + '.hdr', out_fn_hdr])
    

if len(args) < 6:
    err('cut.py [src image or dir] [gdal translate -srcwin parameter 1] [-srcwin param 2] [ -srcwin param 3] # cut image with GDAL and cleanup headers 20220814')
    
A, B, C, D = args[2: 6]
fn = args[1]
if __name__ == "__main__":
    
    files = []
    if args[1][-4:] == '.bin':
        files += [args[1]]
        
    else:
        files += [x.strip() for x in os.popen("ls -1 " + args[1] + os.path.sep + "*.bin").readlines()]
        
    for f in files:
        print(f)
        cut(f, A, B, C, D)
