'''Script for reading primsa data. Tested on L2d 20210725
vnir best fit: r,g,b=37,49,58}

NOrthern hemisphere(N) assumed'''
import os
import sys
import h5py
import math
import numpy as np
sep = os.path.sep
args = sys.argv
pd = sep.join(__file__.split(sep)[:-1]) + sep # present directory

def err(m):
    print("Error: " + str(m)); sys.exit(1)

if len(args) < 2: 
    err("usage:\n\tprisma/read.py [input PRISMA hdf5 file]")

def read_csv(f):
    lines = [x.strip().split(',') for x in open(f).read().strip().split('\n')]
    hdr = lines[0]
    lines = lines[1:]
    dat = {k: [] for k in hdr}
    for line in lines:
        for i in range(0, len(line)):
            dat[hdr[i]].append(line[i])
    #for k in dat: print(k, dat[k])
    return dat

spec = read_csv(pd + "spectral.csv") # print(spec.keys())

def write_hdr(hfn, samples, lines, bands, dsn=None, other=''):
    # WL_VNIR[nm] WL_SWIR[nm] # SWIR_Cube VNIR_Cube
    bands = 1 if bands is None else bands
    rgb_d = None # rgb band indexes from 1
    SWIR = (dsn == 'SWIR_Cube')
    VNIR = (dsn == 'VNIR_Cube')
    rgb_i = None
    if VNIR:  # 630 nm red, 532 nm green, 465 nm blue
        rgb_t, rgb_i, rgb_d = [630, 532, 465], [0,0,0], [None, None, None]
        w_l = spec['WL_VNIR[nm]']
        for i in range(len(w_l)):
            if w_l[i] != '':
                w = int(round(float(w_l[i])))
                di = [abs(rgb_t[j] - w) for j in range(3)]
                for j in range(3):
                    if rgb_d[j] is None or di[j] < rgb_d[j]:
                        rgb_d[j] = di[j]
                        rgb_i[j] = i  # print("better", i, j, w, di[j])
                    else: # i, j, w, rgb_t[j], rgb_i[j], rgb_d[j], di[j]
                        pass
        rgb_i = [x + 1 for x in rgb_i] # 1 index
    w_len = ': ' if (SWIR or VNIR) else ''
    if (SWIR or VNIR):
        w_len += str(int(round(float(spec['WL_SWIR[nm]'][0] if SWIR
                     else spec['WL_VNIR[nm]'][0])))) + 'nm'
    print('+w', hfn)
    rgb_s = (','.join([str(x) for x in rgb_i])) if rgb_i else ''
    lines = ['ENVI' + ('\ndescription = {rgb=' + rgb_s + '}' if VNIR else ''),
             'samples = ' + str(samples),
             'lines = ' + str(lines),
             'bands = ' + str(bands),
             'header offset = 0',
             'file type = ENVI Standard',
             'data type = 4',
             'interleave = bsq',
             'byte order = 0']
    if other != '':
        lines += other.split('\n')
    lines += ['band names = {Band 1' + w_len]
    if bands > 1:
        for i in range(1, bands):
            lines[-1] += ','
            w_len = ': ' if (SWIR or VNIR) else ''
            if(SWIR or VNIR):
                w_len += str(int(round(float(spec['WL_SWIR[nm]'][i] if SWIR else spec['WL_VNIR[nm]'][i])))) + 'nm'
            lines.append('Band ' + str(i + 1) + w_len)
    lines[-1] += '}'
    open(hfn, 'wb').write('\n'.join(lines).encode())

filename = sys.argv[1]
if filename[-3:] != 'he5': err("unexpected filename: .h35 expected")
fn_base = filename[:-4] # print(fn_base) 
datasets, data_sets = [], {}

def iterate(x, s="", parent=None, match=" dataset ", printed=True):
    if printed:
        print(s,x, "attrs:" + str(x.attrs.keys())) # uncomment this line for debug
    keys = None
    try:
        keys = x.keys()
        for k in keys:
            iterate(x[k], s + "**", parent=x, match=match, printed=printed)
    except:
        if len(str(x).split(match)) > 1:
            datasets.append([x, parent])
            dsn = str(x).strip().split(':')[0].split('"')[1].strip('"')
            data_sets[dsn] = [x, parent]
    #if keys is None:
    #   print(s, x)

with h5py.File(filename, "r") as f:
    for k in f.attrs.keys():
        x = f.attrs.get(k)
        print('\t' + str(k) + '=' + str(x), str(type(x)))
    
    def short(x):
        n = 1 # 2
        w = [y[:1] for y in x.split('_')]
        return '_'.join(w)

    map_info = {short(x):f.attrs.get(x)
                for x in ['Projection_Name',
                          'Projection_Id',
                          'Epsg_Code',
                          'Reference_Ellipsoid',
                          'Product_ULcorner_easting',
                          'Product_ULcorner_northing',
                          'Product_LRcorner_easting',
                          'Product_LRcorner_northing']}
    wkt = str(map_info['E_C'])
    wkt = os.popen('gdalsrsinfo -o wkt_esri epsg:' + wkt).read().strip().split()
    wkt = ''.join(wkt)
    wkt = 'coordinate system string = {' + wkt + '}'
    N_S = wkt.split(',')[0].split('Zone')[-1].strip('"')[-1]
    X_size, Y_size = None, None
    iterate(f, printed=False)  # print("fields available:", list(data_sets.keys()))
    want = ['SWIR_Cube', 'VNIR_Cube'] # the meat
    
    if False:  # enable this for more stuff.. examples of avail. stuff
        want += ['Latitude', 'Longitude',
                 'Wgs84_pos_x', 'Wgs84_pos_y', 'Wgs84_pos_z', 
                 'Cw_Swir_Matrix', 'Cw_Vnir_Matrix',
                 'Fwhm_Swir_Matrix', 'Fwhm_Vnir_Matrix']
    # print("fields selected:", str(want))
    for w in want:
        if w not in data_sets:
            err("key not found: " + str(w))

    for dsn in want:  # data set name
        w = data_sets[dsn]
        dsp = str(w[1]).strip().split('"')[1] + '/' + dsn
        dsps = str(dsp)
        dsp = dsp.strip().strip('/').split('/')
        x = f
        for i in dsp:
            x = x[i] # f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Data Fields']['SWIR_Cube'][()]
        x = x[()]
        
        data = np.array(x)
        N = len(data.shape) # how many dimensions? 3 is cube. 2 is 1-band..
        nrow, ncol, nband = None, None, None # image dimensions
        fn = fn_base + '_' + (dsn.replace(' ', '_')) + '.bin'
        hn = fn[:-4] + '.hdr' # print(dsps, '->', fn)
        o_f = open(fn, 'wb')
        dt = '>f4' # default data type to write! always float32, byte order 0
        if N == 3:
            nband = data.shape[1]
            nrow, ncol = data[:,0,:].shape
            for i in range(nband):
                if i % 25 == 0:
                    print("\tband", str(i + 1), 'of', nband)
                data[:,i,:].astype(np.float32).tofile(o_f, '', dt)
        elif N == 2:
            nrow, ncol = data.shape
            data.astype(np.float32).tofile(o_f, '', dt)
        else:
            err('unexpected dimensions')
        if N == 3:
            pass

        X_size = abs((map_info['P_U_e'] - map_info['P_L_e']) / float(ncol))
        Y_size = abs((map_info['P_U_n'] - map_info['P_L_n']) / float(nrow))
        print("X_size", X_size, "Y_size", Y_size)
        ''' map info. Lists geographic information in the following order:
            * Projection name
            * Reference (tie point) pixel x location (in file coordinates)
            * Reference (tie point) pixel y location (in file coordinates)
            * Pixel easting
            * Pixel northing
            * x pixel size
            * y pixel size
            * Projection zone (UTM only)
            * North or South (UTM only)
            * Datum
            * Units
        '''
        m_i = [map_info['P_N'].decode('utf-8'),
               str(1),
               str(1),
               str(map_info['P_U_e']),
               str(map_info['P_U_n']),
               str(X_size),
               str(Y_size),
               map_info['P_I'].decode('utf-8'),
               'North' if N_S == 'N' else 'South',
               map_info['R_E'].decode('utf-8'),
               'units=Meters']
        mapinfo = 'map info = {' + ','.join(m_i) + '}'
        other = mapinfo + '\n' + wkt
        write_hdr(hn, ncol, nrow, nband, dsn, other=other)
        o_f.close()
