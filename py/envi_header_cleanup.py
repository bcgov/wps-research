'''Clean up envi header so that they can be opened in IMV

20220514: 
    default:
        all hdr in present folder. Process in parallel!'''
from misc import *

if len(args) < 2:
    # err("python3 envi_header_cleanup.py [input envi header filename .hdr]")
    lines = [x.strip() for x in os.popen('ls -1 *.hdr').readlines()]
    found = False
    jobs = []
    for line in lines:
        if exist(line):
            c = 'python3 ' + __file__ + ' ' + line
            jobs.append(c)
            found = True
    if not found:
        err("file not found: " + args[1])

    parfor(run, jobs, 8)
    sys.exit(0)

data = open(args[1]).read().strip()
n_band_names, in_band_names, nb = 0, False, 0
data = data.replace("description = {\n", "description = {")
data = data.replace("band names = {\n", "band names = {")
lines, non_bandname_lines = data.split("\n"), []
bandname_lines = []

for i in range(0, len(lines)):
    line = lines[i].strip()

    w = [x.strip() for x in line.split("=")]
    if len(w) > 1:
        if w[0].strip() == 'bands':
            nb = int(w[1].strip())
        lines[i] = ' = '.join([x.strip() for x in w])
    line = lines[i].strip()

    if len(line.split("band names")) > 1:
        in_band_names = True
    # print(line + (" TRUE" if in_band_names else ""))

    if in_band_names:
        n_band_names += 1
        if len(line.split("}")) < 2:
            w = line.split(',')
            line = ''.join(w[:-1]) + ',' + w[-1]
            lines[i] = line
        else: # on last band names line:
            lines[i] = line.replace(',', '')

    if in_band_names:
        # print("*", line)
        bandname_lines.append(line) # track band names we have
    else:
        non_bandname_lines.append(line) # record non-band-name lines,
        # in case we need to fill the band-names in

    if in_band_names:
        if len(line.split("}")) > 1:
            in_band_names = False

if nb != n_band_names:
    if n_band_names > nb:
        # print("n_band_names", n_band_names, "nb", nb)
        bandname_lines = bandname_lines[:nb]
        bandname_lines[-1] = bandname_lines[-1].strip() + "}"
    if n_band_names > 0 and n_band_names < nb:
        bandname_lines[-1] = bandname_lines[-1].strip().strip('}')
        for i in range(1, nb + 1):
            if i > n_band_names:
                bandname_lines[-1] = bandname_lines[-1].strip().strip("}").strip(",") + ','
                pre = "band names = {" if i == 1 else ""
                bandname_lines.append(pre + "Band " + str(i) + ",")
        bandname_lines[-1] = bandname_lines[-1].strip().strip(',') + "}"

    if n_band_names == 0:
        bandname_lines.append("band names = {Band 1,")
        for i in range(1, nb):
            bandname_lines.append("Band " + str(i + 1) + ",")
        bandname_lines[-1] = bandname_lines[-1].strip().strip(",") + "}"

bandname_lines[-1] = bandname_lines[-1].replace(',', '') # no comma in last band names record
lines = non_bandname_lines + bandname_lines
data = ('\n'.join(lines)).strip()

# print(data)
open(args[1] + '.bak', 'wb').write(open(args[1]).read().encode())
open(args[1], 'wb').write(data.encode())

# now trim the band names strings
band_names = [x.strip() for x in os.popen("python3 ~/GitHub/wps-research/py/envi_header_band_names.py " + args[1]).readlines()]
samples, lines, bands = read_hdr(args[1])
run(['python3 ~/GitHub/wps-research/py/envi_header_modify.py', 
     args[1],
     lines,
     samples, 
     bands] +
    band_names + ['1'])
