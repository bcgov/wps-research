'''rasterize a target file (one hot encoding)..tgt file of the format used in the MVP software'''

from misc import * 

if len(args) < 3:
    err("python3 vector_target_rasterization.py [input target file] [input raster file)]")
    # python3 vector_target_rasterization.py stack.bin_targets.csv stack.bin

tfn, rfn = args[1], args[2]
hfn = hdr_fn(rfn) # raster header file name

nc, nr, nb = read_hdr(hfn)



