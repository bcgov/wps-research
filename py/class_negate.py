# 20200610 invert a float-valued binary class map
# require the values be 0 or 1

from misc import *

if len(args) < 2:
    err("class_negate [input binary file]")

fn = args[1]
samples, lines, bands, data = read_binary(fn)

data = np.array([1. - d for d in data])

write_binary(data, args[1] + "_negate.bin")
