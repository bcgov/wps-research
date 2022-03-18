''' 
within present hierarchy, find Sentinel2 data in expected format
- stack multiple overlapping frames by co-registering the other frames,
onto the selected reference footprint
- reference footprint could be a sentinel2 frame OR something else (an AOI)


'''
from misc import *
if len(args) < 2:
    err('python3 sentinel2_stack_multiscene.py [AOI footprint ENVI format .bin]')

rf = args[1]
if rf[-4:] != '.bin':
    err('reference footprint must be raster (extension .bin) ENVI format')

# check if reference footprint is a Sentinel2 tile.
ref_tile = None
try:
    w = rf.split(sep)[-2]
    w = w.split('_')
    ref_tile = w[5]
    print(ref_tile)
except:
    pass

print('reference tile ID=', ref_tile)

coreg_files = []
for f in [x.strip() for x in os.popen('find ./ -name "SENTINEL2_L2A_EPSG*10m.bin"').readlines()]:
    parent_path = sep.join(os.path.abspath(f).split(sep)[:-1])
    parent_f = parent_path.split(sep)[-1]

    # is one-up/parent folder: an S2 folder i.e. S2*.SAFE?
    if parent_f[-5:] == '.SAFE' and parent_f[:2] == 'S2':
        w = parent_f.split('_')
        if w[6][8] != 'T':
            err('unexpected folder name format: ' + parent_path)
        ds, ts = w[6][:8], w[6][9: 15]

        coreg_f = f
        if ref_tile is not None and ref_tile == w[5]:
            print("Don't need coreg:")
        else:
            coreg_f = f + '_coreg.bin'

        print(w, ds, ts, coreg_f)
        coreg_files.append([ds + ts, coreg_f, str(file_size(f) / (1024. * 1024.))+ 'MB'])

coreg_files.sort()
for f in coreg_files:
    print(f)


