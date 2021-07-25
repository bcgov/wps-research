'''script for reading primsa data. in progress.'''
import numpy as np
import h5py
import sys
import os

filename = sys.argv[1]

def iterate(x, s=""):
    print(s,x)
    keys = None
    try:
        keys = x.keys()
        for k in keys:
            iterate(x[k], s + "**")
    except:
        pass

    #if keys is None:
    #    print(s, x)

with h5py.File(filename, "r") as f:
    # List all groups
    iterate(f)
    
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
