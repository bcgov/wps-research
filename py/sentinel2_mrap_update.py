'''20250608 sentinel2_mrap_update.py
'''

from misc import err, run, sep, hdr_fn
import os

def list_bins():
    bin_files = os.popen('ls -1 *mrap.bin').read().strip().split('\n')
    bins = []
    for b in bin_files:
        if os.path.exists(b):
            bins += [b]
    return set(bins)

bin_files = list_bins()

run('sync_recent.py')  # refresh L2 data
run('sentinel2_extract_cloudfree_swir_nir.py')
run('sentinel2_mrap.py')
run('sentinel2_mrap_merge.py')
run('clean')
run('rm *.vrt *.xml')

bin_files_new = list_bins()
bin_files_added = bin_files_new - bin_files

for b in bin_files_added:
    date_s = b.split('_')[0]

    bin_dir = date_s
    
    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)

    os.symlink(b, bin_dir + sep + b)
    os.symlink(hdr_fn(b), bin_dir + sep + hdr_fn(b))
    
    small_dir = 'small_' + date_s

    if not os.path.exists(small_dir):
        os.mkdir(small_dir)

    run('raster_warp_all -s 2 ' + bin_dir + ' ' + small_dir)

    gz_file = small_dir + '.tar.gz'

    if not os.path.exists(gz_file):
        run('tar cvfz ' + gz_file + ' ' + small_dir)
