'''assuming we are 
A) inside a .dim folder from snap for compact-pol data..or
B) inside a C2 folder for compact-pol data..(PolSARPro format already)

apply Dey et al [1] mf3cc compact-pol decom, C implementation'''
import os
import sys
args = sys.argv
sep = os.path.sep

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    print(c); a = os.system(c)
    if a != 0: err("command failed:\n\t" + c)

nwin = 9
c2_start = False
if not os.path.exists('C11.hdr'):
    print("snap_m3fcc # optional param: window size: default 9 # needs to be run from inside a .dim folder for compact-pol data")
    if len(args) > 1:
        nwin = int(args[1])

    ''' input files:'''
    ins = ['i_RCH2.img', 'i_RCH.img', 'q_RCH2.img', 'q_RCH.img']
    for i in ins:
        if not os.path.exists(i):
            err("req'd input file: " + i)
    if not os.path.exists("i_RCH.hdr"):
        err("req'd input file: i_RCH.hdr")

    # swap byte order from european to american convention
    run("snap2psp ./")

    '''output files from snap2psp'''
    bins = ['i_RCH2.bin',  'i_RCH.bin', 'q_RCH2.bin', 'q_RCH.bin']

    # get image dims, will need this to generate C2 matrix cf [2]
    nrow, ncol = None, None
    lines = [x.strip() for x in open('i_RCH.hdr').readlines()]
    for line in lines:
        w = [x.strip() for x in line.split("=")]
        if w[0] == 'samples': ncol = int(w[1])
        if w[0] == 'lines': nrow = int(w[1])
    print("nrow", nrow, "ncol", ncol)

    run('convert_iq_to_cplx i_RCH.bin q_RCH.bin ch.bin') # convert to envi type6 data
    run('convert_iq_to_cplx i_RCH2.bin q_RCH2.bin cv.bin')

    run(' '.join(['cp_2_t2', str(nrow), str(ncol), 'ch.bin', 'cv.bin'])) # convert to C2 mtx cf [1]

    run('mv T11.bin C11.bin; mv T22.bin C22.bin') # use expected filenames
    run('mv T12_real.bin C12_real.bin; mv T12_imag.bin C12_imag.bin')

    run('eh2cfg ch.bin.hdr') # make sure there is config.txt
else:
    c2_start = True

# if there is a C11, etc. we can just jump right in
run('mf3cc ./ ' + str(nwin)) # run the decom

run('cat Pd_CP.bin Pv_CP.bin Ps_CP.bin Theta_CP.bin > stack.bin') # stack things

lines = ['ENVI', # write a header for the stack
         'samples = ' + str(ncol),
         'lines = ' + str(nrow),
         'bands = 4',
         'header offset = 0',
         'file type = ENVI Standard',
         'data type = 4',
         'interleave = bsq',
         'byte order = 0',
         'band names = {Pd_CP.bin,',
         'Pv_CP.bin,',
         'Ps_CP.bin,',
         'Theta_CP.bin}']
open('stack.hdr', 'wb').write(('\n'.join(lines)).encode())

# clean up some intermediary files
d = ['C11.bin', 'C12_imag.bin', 'C12_real.bin', 'C22.bin'] if not c2_start else []

d += ['ch.bin', 'cv.bin',
     'i_RCH2.bin', 'i_RCH.bin',
     'Pd_CP.bin', 'Ps_CP.bin', 'Pv_CP.bin',
     'q_RCH2.bin', 'q_RCH.bin', 'Theta_CP.bin']
for i in d:
    run('rm ' + i)

'''
[1] Dey et al S. Dey, A. Bhattacharya, D. Ratha, D. Mandal and A. C. Frery, "Target Characterization and Scattering Power Decomposition for Full and Compact Polarimetric SAR Data," in IEEE Transactions on Geoscience and Remote Sensing, doi: https://doi.org/10.1109/TGRS.2020.3010840.
[2] Cloude et al "Compact Decomposition Theory" IEEE GRSL 2011'''
