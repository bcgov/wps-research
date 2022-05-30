''' 20220317 within present hierarchy, find Sentinel2 data in expected format
- stack multiple overlapping frames by co-registering the other frames,
onto the selected reference footprint
- reference footprint could be a sentinel2 frame OR something else (an AOI) e.g.:
python3 ~/GitHub/wps-research/py/sentinel2_stack_multiscene.py ./T10UGA/S2A_MSIL2A_20210721T185921_N0301_R013_T10UGA_20210721T230008.SAFE/SENTINEL2_L2A_EPSG_32610_10m.bin
'''
from misc import *
if len(args) < 2:
    err('python3 sentinel2_stack_multiscene.py [AOI footprint ENVI-type .bin]')

rf = args[1]
if rf[-4:] != '.bin':
    err('reference footprint must be raster (extension .bin) ENVI format')
samples, lines, bands = read_hdr(hdr_fn(rf))

ref_tile = None  # check if reference footprint is Sentinel2 tile
try:
    w = rf.split(sep)[-2].split('_')
    ref_tile = w[5]
    print(ref_tile)
except:
    pass

print('reference tile ID=', ref_tile)
coreg_files = []
for f in [x.strip() for x in os.popen('find ./ -name "SENTINEL2_L2A_EPSG*10m.bin"').readlines()]:
    ff = f
    hff = ff[:-4] + '.hdr'
    f = f + '_active.bin'
    hf = f[:-4] + '.hdr'

    # active fire detection result needs map info for coreg step
    X = get_map_info_lines_idx(hf)
    if X[0] is None or X[1] is None:
        cmd = ' '.join(['python3',
                        pd + 'envi_header_copy_mapinfo.py',
                        hff,
                        hf])
        run(cmd)  # run if either map info field in target is blank
    parent_path = sep.join(os.path.abspath(f).split(sep)[:-1])
    parent_f = parent_path.split(sep)[-1]

    # is one-up/parent folder: an S2 folder i.e. S2*.SAFE?
    if parent_f[-5:] == '.SAFE' and parent_f[:2] == 'S2':
        w = parent_f.split('_')
        if w[6][8] != 'T':
            err('unexpected folder name format: ' + parent_path)
        ds, ts = w[6][:8], w[6][9: 15]

        if not exists(f):
            err('file does not exist:' + f)

        coreg_f = f
        need_coreg = False
        if ref_tile is not None and ref_tile == w[5]:
            pass # print("Don't need coreg:")
        else:
            coreg_f = f + '_coreg.bin'
            need_coreg = True

        print(w, ds, ts, coreg_f)
        coreg_files.append([ds + ts, coreg_f, str(file_size(f) / (1024. * 1024.))+ 'MB', need_coreg])

print("-----------------------")
coreg_files.sort()
for f in coreg_files:
    print(f)
    if f[-1]:
        corg_fn = f[-3]
        orig_fn = corg_fn[:-10]
        cmd = ' '.join(['python3',
                        pd + 'raster_project_onto.py',
                        orig_fn,
                        rf,
                        corg_fn])
        if not exists(corg_fn):
            run(cmd) # should run a few in parallel?

files = [coreg_files[i][1] for i in range(len(coreg_files))]
cmd = 'cat ' + (' '.join(files)) + ' > raster.bin'
if not exists('raster.bin'):
    run(cmd)

if not exists('raster.hdr'):
    write_hdr('raster.hdr', samples, lines, len(coreg_files), files)
    run(['python3 ' + pd + 'envi_header_copy_mapinfo.py',
         hdr_fn(rf),
         'raster.hdr'])
