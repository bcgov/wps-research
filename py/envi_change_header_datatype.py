# given a series of files ending in *output.hdr, change the type string from 5 to 4

import os
import sys

files = os.popen("ls -1 *output.hdr").readlines()
files = [f.strip() for f in files]

for f in files:
    f = f.strip()
    print(f)
    nfn = f.replace("output.hdr", "output4.hdr")
    
    lines = open(f).readlines()
    lines = [line.strip() for line in lines]
    
    for i in range(0, len(lines)):
        if lines[i] == "data type = 5":
            lines[i] = "data type = 4"
    print("+w", nfn)
    open(nfn, "wb").write(("\n".join(lines)).encode())
