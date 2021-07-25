'''Script for reading primsa data. Tested on L2d 20210725'''
import numpy as np
import h5py
import sys
import os
sep = os.path.sep
pd = sep.join(__file__.split(sep)[:-1]) + sep # present directory

def err(m):
    print("Error: " + str(m)); sys.exit(1)

def read_csv(f):
    lines = [x.strip().split(',') for x in open(f).read().strip().split('\n')]
    hdr = lines[0]
    lines = lines[1:]
    dat = {k: [] for k in hdr}
    for line in lines:
        for i in range(0, len(line)):
            dat[hdr[i]].append(line[i])
    #for k in dat:
    #   print(k, dat[k])
    return dat

spec = read_csv(pd + "spectral.csv")

print(spec.keys())
def write_hdr(hfn, samples, lines, bands, dsn=None):
    # WL_VNIR[nm] WL_SWIR[nm] # SWIR_Cube VNIR_Cube
    bands = 1 if bands is None else bands
    
    SWIR = (dsn == 'SWIR_Cube')
    VNIR = (dsn == 'VNIR_Cube')

    w_len = ': ' if (SWIR or VNIR) else ''
    if (SWIR or VNIR):
        w_len += str(int(round(float(spec['WL_SWIR[nm]'][0] if SWIR else spec['WL_VNIR[nm]'][0])))) + ' nm'

    print('+w', hfn)
    lines = ['ENVI',
             'samples = ' + str(samples),
             'lines = ' + str(lines),
             'bands = ' + str(bands),
             'header offset = 0',
             'file type = ENVI Standard',
             'data type = 4',
             'interleave = bsq',
             'byte order = 0',
             'band names = {Band 1' + w_len]
    if bands > 1:
        for i in range(1, bands):
            lines[-1] += ','
            w_len = ': ' if (SWIR or VNIR) else ''
            if(SWIR or VNIR):
                w_len += str(int(round(float(spec['WL_SWIR[nm]'][i] if SWIR else spec['WL_VNIR[nm]'][i])))) + ' nm'
            lines.append('Band ' + str(i + 1) + w_len)
    lines[-1] += '}'
    open(hfn, 'wb').write('\n'.join(lines).encode())

filename = sys.argv[1]
if filename[-3:] != 'he5': err("unexpected filename")
fn_base = filename[:-4] # print(fn_base) 
datasets, data_sets = [], {}

def iterate(x, s="", parent=None):
    # print(s,x) # uncomment this line for debug
    keys = None
    try:
        keys = x.keys()
        for k in keys:
            iterate(x[k], s + "**", parent=x)
    except:
        if len(str(x).split(" dataset ")) > 1:
            datasets.append([x, parent])
            dsn = str(x).strip().split(':')[0].split('"')[1].strip('"')
            data_sets[dsn] = [x, parent]
    #if keys is None:
    #   print(s, x)

with h5py.File(filename, "r") as f:
    iterate(f)  # list all groups
    print("fields available:", list(data_sets.keys()))
    # swaths = f['HDFEOS']['SWATHS']
    want = ['SWIR_Cube', 'VNIR_Cube', 'Latitude', 'Longitude',
            'Cw_Swir_Matrix', 'Cw_Vnir_Matrix', 'Fwhm_Swir_Matrix', 'Fwhm_Vnir_Matrix']
    print("fields selected:", str(want))
    for w in want:
        if w not in data_sets: err("key not found: " + str(w))

    for dsn in want:  # data set name
        w = data_sets[dsn]
        dsp = str(w[1]).strip().split('"')[1] + '/' + dsn
        dsps = str(dsp)
        dsp = dsp.strip().strip('/').split('/')
        
        # print("dsn", dsn) # SWIR_Cube ['HDFEOS', 'SWATHS', 'PRS_L2D_HCO', 'Data Fields', 'SWIR_Cube']
        x = f
        for i in dsp:
            x = x[i] # f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Data Fields']['SWIR_Cube'][()]
        x = x[()]
        
        data = np.array(x)
        N = len(data.shape) # how many dimensions? 3 is cube. 2 is 1-band..
        nrow, ncol, nband = None, None, None # image dimensions
        fn = fn_base + '_' + (dsn.replace(' ', '_')) + '.bin'
        hn = fn[:-4] + '.hdr'
        print(dsps, '->', fn) # print('\t', dsn, dsp, data.shape)
        o_f = open(fn, 'wb')
        dt = '>f4' # default data type to write! always float32, byte order 0
        if N == 3:
            nband = data.shape[1]
            nrow, ncol = data[:,0,:].shape
            for i in range(nband):
                # print("\twriting band", str(i + 1))
                data[:,i,:].astype(np.float32).tofile(o_f, '', dt)
        elif N == 2:
            nrow, ncol = data.shape
            data.astype(np.float32).tofile(o_f, '', dt)
        else:
            err('unexpected dimensions')
        if N == 3:
            pass
            # print("\tclosing file..")
        o_f.close()
        write_hdr(hn, ncol, nrow, nband, dsn)
