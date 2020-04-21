import os
import sys

a = os.system("mkdir new")

files = os.popen("ls -1 *.bin").readlines()
files = [f.strip() for f in files]

for f in files:
    f = f.strip()
    hf = f[:-3] + 'hdr'

    hf2 = "new" + os.path.sep + hf


    lines = open(hf).readlines()
    lines = [line.strip() for line in lines]

    for i in range(0, len(lines)):
        line = lines[i]
        if line == "bands   = 13":
            lines[i] = "bands   = 11"

    #print("------------------------")
    print(hf2)
    #for line in lines:
    #    print("  " + line)


    del lines[-1]
    del lines[-2]
    lines[-1] = lines[-1].replace(",", "}")

    #print("**")
    #for line in lines:
    #    print("  " + line)

    print(hf2)
    open(hf2, "wb").write(("\n".join(lines)).encode())

    print("write binary file using c++ program..")

    if not os.path.exists("rm_band"):
        a = os.system("g++ misc.cpp rm_band.cpp -o rm_band")
    cmd = "./rm_band " + f + " new" + os.path.sep + f + " 11 13"
    print(cmd)
    a = os.system(cmd)