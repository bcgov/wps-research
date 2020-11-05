# from two envi header files (given), produce a new header file reflecting the bands from both header files being combined
import os
import sys
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    print(c)
    a = os.system(c)
    if a != 0: err("failed to run: " + str(c))

if len(args) < 4:
    err("envi_header_cat.py [.hdr file #1] " +
        "[.hdr file #2] [output .hdr file] #" +
        "[optional prefix for bandnanes from .hdr file #1] " +
        "[optional prefix for bandnames from .hdr file #2]")

pre1, pre2 = '', ''

if len(args) > 4:
    pre1 = args[4]

if len(args) > 5:
    pre2 = args[5]

def get_band_names_lines(data):
    band_name_lines, in_band_names = [], False
    lines = [x.strip() for x in data.strip().split("\n")]
    for i in range(0, len(lines)):
        if len(lines[i].split("band names =")) > 1:
            in_band_names = True

        if in_band_names:
            # print(lines[i])
            band_name_lines.append(lines[i])
            if len(lines[i].split("}")) > 1:
                in_band_names = False
    return band_name_lines

if not exists(args[1]) or not exists(args[2]):
    err("please check input files:\n\t" + args[1] + "\n\t" + args[2])

run('python3 ' + pd + 'envi_header_cleanup.py ' + args[1])
run('python3 ' + pd + 'envi_header_cleanup.py ' + args[2]) # should really call directly, whatever

i_dat, o_dat = open(args[1]).read(),  open(args[2]).read()
bn_1, bn_2 = get_band_names_lines(i_dat), get_band_names_lines(o_dat)
lines1 , lines2 = i_dat.strip().split('\n'), o_dat.strip().split('\n')

band_count = len(bn_1) + len(bn_2) # add band counts
print("band_count", band_count)

if lines2[-1] not in bn_2:
    print("unexpected header formatting")

lines2[-1] = lines2[-1].strip().strip('}') + ','
bn_1[0] = bn_1[0].split('{')[1]
lines2 = lines2 + bn_1

for i in range(len(lines2)):
    if len(lines2[i].split('bands   =')) > 1:
        lines2[i] = lines2[i].split('=')[0] + '= ' + str(band_count)

    if len(lines2[i].split("description =")) > 1:
        lines2[i] = "description = {" + args[3][:-4] + '.bin}'

print("")
for line in lines2:
    print(line)

f = open(args[3], "wb")
if not f:
    err("failed to open output file: " + args[3])

f.write('\n'.join(lines2).encode())
f.close()
