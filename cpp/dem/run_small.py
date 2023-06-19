import os
import sys

def run(c):
    a = os.system(c)
    return a

run("make")
run("raster_multilook dat/stack.bin 4")
run("raster_scale dat/stack.bin_mlk.bin")
run("envi_header_copy_bandnames.py dat/stack.hdr dat/stack.bin_mlk.hdr")
run("./dem dat/stack.bin_mlk.bin_scale.bin")
