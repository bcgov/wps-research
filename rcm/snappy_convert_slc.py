''' given a collection of unzipped RCM Quad-pol data: for each SLC folder:
        1) convert to PolSARPro format ("i/q format"
        2) convert i/q format to s2 "Scattering Matrix" format
        3) apply scm method (a visualization method of Shane Cloude)

references:
    [1] https://senbox.atlassian.net/wiki/spaces/SNAP/pages/19300362/How+to+use+the+SNAP+API+from+Python


Note: might need to run from /home/$USER.snap/snap-python/snappy/ folder
unless you know how to set the various paths

export PYTHONPATH=$PYTHONPATH:/home/$USER/.snap/snap-python/
'''
import os # usual python stuff
import sys
import snappy # ESA SNAP stuff
import shutil
from snappy import ProductIO
args = sys.argv
sep = os.path.sep
# sys.path.append('/home/' + os.popen('whoami').read().strip() + '/.snap/snap-python/')

def err(m):
    print('Error:', m); sys.exit(1)

def run(c):
    print(c)
    a = os.system(c)
    if a != 0: err(c)
    return a

def hdr_fn(bin_fn):
    # return filename for header file, given name for bin file
    hfn = bin_fn[:-4] + '.hdr'
    if not os.path.exists(hfn):
        hfn2 = bin_fn + '.hdr'
        if not os.path.exists(hfn2):
            err("didn't find header file at: " + hfn + " or: " + hfn2)
        return hfn2
    return hfn

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

def md(d):  # make folder if not yet exist
    if not os.path.exists(d): os.mkdir(d)

exe = "/home/" + \
      os.popen("whoami").read().strip() + \
      "/GitHub/wps-research/cpp/convert_iq_to_s2.exe"
exe = os.path.abspath(exe)
print(exe)

data_folder = args[1] if len(args) > 1 else os.path.abspath(os.getcwd()) + os.path.sep

if not os.path.exists(exe):
    print("Error: failed to find convert_iq_to_s2.exe"); sys.exit(1)

cmd = "ls -1d " + os.path.abspath(data_folder) + os.path.sep + "*_SLC" 
files = [f.strip() for f in os.popen(cmd).readlines()]
folders = [f + '/' for f in files]

print(folders)
for i in range(0, len(folders)):
    f = folders[i] + 'manifest.safe'
    print('\n  ' + f)

    out_folder = folders[i] + 'PolSARPro'; md(out_folder)
    s2_folder = folders[i]  + 's2'; md(s2_folder)
    scm_folder = folders[i] + 'scm'; md(scm_folder)

    d = ProductIO.readProduct(f)  # read SLC data 
    ProductIO.writeProduct(d, out_folder, 'PolSARPro')  # convert to PolSARPro format

    # check for type2 data and convert to type4

    x = out_folder + sep + 'i_HH.bin'
    xhf = hdr_fn(x)  # locate header file
    lines = [x.strip() for x in open(xhf).read().strip().split('\n')]
    data_type = None
    for line in lines:
        if len(line.split('data type')) > 1:
            data_type = int(line.split('=')[1].strip())
            if data_type != 4:
                for bf in ['i_HH.bin', 'i_HV.bin', 'i_VH.bin', 'i_VV.bin',
                           'q_HH.bin', 'q_HV.bin', 'q_VH.bin', 'q_VV.bin']:
                    tf = out_folder + sep + 'tmp.bin'
                    cv = "/home/" + os.popen("whoami").read().strip() + "/GitHub/wps-research/cpp/cv.exe"
                    run(' '.join([cv,
                                  out_folder + sep + bf,
                                  tf,
                                  str(4)]))  # Usage: cv [input file name] [output file name] [output data type]
                    shutil.move(tf, out_folder+sep+bf)
                    mod = [x.rstrip() for x in open(hdr_fn(out_folder+sep+bf)).read().strip().split('\n')]
                    for j in range(len(mod)):
                        if len(mod[j].split('data type =')) > 1:
                            mod[j] = mod[j].replace('data type = ' + str(data_type), 'data type = 4')
                    open(hdr_fn(out_folder + sep + bf), 'wb').write(('\n'.join(mod)).encode())
                    # sys.exit(1)
    run(exe + " " + out_folder + " " + s2_folder)
    a = os.system(' '.join(['scm',  # a visualization method of Shane Cloude
                            s2_folder, # input s2 matrix format data
                            scm_folder, # output for visualization result
                            'box', # filter type used for visualization
                            '5', # filter window size
                            'yes', # correct for average faraday rotation on scene
                            '1', # alpha angle used for visualization
                            '3'])) # vertical multiook factor

    f1 = folders[i] + 'scm_fr'
    f2 = folders[i] + 'scm_fr_t4'
    f3 = folders[i] + 'scm_fr_t4_fl'
    for fi in [f1, f2, f3]:
        cmd = 'rm -rf ' + fi
        print(cmd)
        a = os.system(cmd)
