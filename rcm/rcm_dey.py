# convert to C2 to run decoms

def read_hdr(hdr):
    samples, lines, bands = 0, 0, 0
    for line in open(hdr).readlines():
        line = line.strip()
        words = line.split('=')
        if len(words) == 2:
            f, g = words[0].strip(), words[1].strip()
            if f == 'samples':
                samples = g
            if f == 'lines':
                lines = g
            if f == 'bands':
                bands = g
    return samples, lines, bands

import os
sep = os.path.sep
files = [x.strip() for x in os.popen('ls -1 *.bin').readlines()]

nc, nr, nb = None, None, None

for f in files:  # print(f)
    d = f[:-4] + sep
    print(d)

    if not os.path.exists(d): os.mkdir(d)
    a = os.system('~/GitHub/wps-research/cpp/unstack.exe ' + f)
    ci = ['C11.bin',
          'C22.bin',
          'C12_real.bin',
          'C12_imag.bin']
    
    for i in range(1, 5):
        fi = f + '_' + str(i).zfill(3) + '.bin'
        hi = fi[:-4] + '.hdr'
        print(fi)
        print(hi)

        if nc is None:
            nc, nr, nb = read_hdr(hi)
        # c = 'mv ' + fi + ' ' + d + ci[i-1]
        c = 'boxcar ' + fi + ' ' + nr + ' ' + nc + ' ' + str(7) + ' ' + d + ci[i-1]
        print('\t', c)
        a = os.system(c)


        if i == 1:
            c = 'mv ' + hi + ' ' + d + ci[i-1][:-4] + '.hdr'
            print('\t', c)
            a = os.system(c)
    
    a = os.system("rm *.bin_*.bin")
