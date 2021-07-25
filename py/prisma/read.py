'''script for reading primsa data. in progress.'''
import numpy as np
import h5py
import sys
import os

def err(m):
    print("Error: " + str(m)); sys.exit(1)

def write_hdr(hfn, samples, lines, bands):
    print('+w', hfn)
    lines = ['ENVI',
             'samples = ' + str(samples),
             'lines = ' + str(lines),
             'bands = ' + str(bands),
             'header offset = 0',
             'file type = ENVI Standard',
             'data type = 4',
             'interleave = bsq',
             'byte order = 0']
    
    lines.append('band names = {Band 1') # }')
    if bands > 1:
        for i in range(1, bands):
            lines[-1] += ','
            lines.append('Band ' + str(i + 1))
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
        
        # print("dsp", dsp) # SWIR_Cube ['HDFEOS', 'SWATHS', 'PRS_L2D_HCO', 'Data Fields', 'SWIR_Cube']
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
        write_hdr(hn, ncol, nrow, nband)

    '''


    sys.exit(1)

    y = np.array(f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Data Fields']['SWIR_Cube'][()])
    nband = y.shape[1]
    nrow, ncol = y[:,0,:].shape
    print("nrow", nrow, "ncol", ncol, "nband", nband)

    dt = '>f4'
    o_f = open("swir.bin", "wb")
    for i in range(nband):
        print(i)
        y[:,i,:].astype(np.float32).tofile(o_f, '', dt)
    o_f.close()
    '''

    #print(f['Info']['Header']['FrameNumber']) # keys())
'''
Keys: <KeysViewHDF5 ['HDFEOS', 'HDFEOS INFORMATION', 'Info', 'KDP_AUX']>
HDFEOS -> <KeysViewHDF5 ['ADDITIONAL', 'SWATHS']>
HDFEOS INFORMATION -> <KeysViewHDF5 ['StructMetadata.0']>
Info -> <KeysViewHDF5 ['Ancillary', 'Header', 'Housekeeping']>
KDP_AUX -> <KeysViewHDF5 ['Cw_Swir_Matrix', 'Cw_Vnir_Matrix', 'Fwhm_Swir_Matrix', 'Fwhm_Vnir_Matrix', 'LOS_Pan', 'LOS_Swir', 'LOS_Vnir']>
'''

# type "<u4"  : 4 byte unsigned float, little endian..

'''
________ <HDF5 group "/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields" (8 members)>
__________ <HDF5 dataset "Cloud_Mask": shape (1000, 1000), type "|u1">
__________ <HDF5 dataset "FrameNumber": shape (1000,), type "<u4">
__________ <HDF5 dataset "LandCover_Mask": shape (1000, 1000), type "|u1">
__________ <HDF5 dataset "SWIR_Cube": shape (1000, 173, 1000), type "<u2">
__________ <HDF5 dataset "SWIR_PIXEL_SAT_ERR_MATRIX": shape (1000, 173, 1000), type "|u1">
__________ <HDF5 dataset "SunGlint_Mask": shape (1000, 1000), type "|u1">
__________ <HDF5 dataset "VNIR_Cube": shape (1000, 66, 1000), type "<u2">
__________ <HDF5 dataset "VNIR_PIXEL_SAT_ERR_MATRIX": shape (1000, 66, 1000), type "|u1">
'''
