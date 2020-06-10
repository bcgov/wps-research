# 20200610 invert a float-valued binary class map
# require the values be 0 or 1

from misc import *

if len(args) < 2:
    err("class_negate [input binary file]")

fn = args[1]
samples, lines, bands, data = read_binary(fn)

count = {}
for d in data:
    if d not in count:
        count[d] = 0
    count[d] += 1

print(count.keys())

