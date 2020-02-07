# quick stats on float file
from misc import * 

if len(args) < 2:
    err("qs [input binary file]")

fn = args[1]
samples, lines, bands, data = read_binary(fn)

d = np.array(data)

print("min", np.min(d))
print("max", np.max(d))
print("std", np.std(d))
