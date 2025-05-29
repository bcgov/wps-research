''' 20221105 raster_histogram_mask.py

Run raster_histogram.py twice:
    on portion under mask (0)
    and portion not under mask (0)
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from misc import assert_exists, hdr_fn, read_hdr, read_float, band_names, err

def raster_histogram_with_mask(input_file, mask_file, band_select=[0, 1, 2], histogram_scaling_factor=None):
    assert_exists(input_file)
    assert_exists(mask_file)

    hdr = hdr_fn(input_file)
    mask_hdr = hdr_fn(mask_file)

    [samples, lines, bands] = [int(x) for x in read_hdr(hdr)]
    npx = lines * samples
    data = read_float(input_file).reshape((bands, npx))

    mask_shape = [int(x) for x in read_hdr(mask_hdr)]
    assert mask_shape[0] == samples and mask_shape[1] == lines, "Mask dimensions must match image dimensions"

    mask = read_float(mask_file).reshape(npx)
    mask = np.nan_to_num(mask, nan=0).astype(bool)

    try:
        bn = band_names(hdr)
    except:
        bn = [f"Band {i}" for i in range(bands)]

    N = len(band_select)
    rng = range(N)
    dat = [data[band_select[i],] for i in rng]

    if histogram_scaling_factor is not None:
        from view import scale
        for i in rng:
            dat[i] = scale(np.array(dat[i]), True, True)

    under_mask = [np.array(dat[i][mask & ~np.isnan(dat[i])]) for i in rng]
    outside_mask = [np.array(dat[i][~mask & ~np.isnan(dat[i])]) for i in rng]

    all_vals = np.hstack([np.hstack((under_mask[i], outside_mask[i])) for i in rng])
    my_min, my_max = np.nanmin(all_vals), np.nanmax(all_vals)
    n_bins = 5000
    bins = np.linspace(my_min, my_max, n_bins)

    plt.figure()
    plt.title(f'Histograms with mask for {input_file}')
    col = ['r', 'g', 'b']

    for i in rng:
        label_masked = f"{bn[band_select[i]]} (masked)"
        label_unmasked = f"{bn[band_select[i]]} (outside)"
        plt.hist(under_mask[i], bins=bins, histtype='step', label=label_masked,
                 color=(col[i] if i < 3 else None), linestyle='solid')
        plt.hist(outside_mask[i], bins=bins, histtype='step', label=label_unmasked,
                 color=(col[i] if i < 3 else None), linestyle='dashed')

    plt.legend()
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        err('Usage:\n\traster_histogram_with_mask.py [input_file] [mask_file] '
            '[optional: red_band_idx] [optional: green_band_idx] [optional: blue_band_idx]\n'
            'Note: Band indices are 1-based (e.g., 1 2 3)')

    fn, mask_fn = args[1], args[2]
    band_select = [0, 1, 2]  # default bands

    if len(args) >= 6:
        try:
            band_select = [int(args[3])-1, int(args[4])-1, int(args[5])-1]
        except:
            err("Invalid band indices. Use integers >= 1.")
    elif len(args) > 3:
        try:
            band_select = [int(b)-1 for b in args[3:]]
        except:
            err("Invalid band indices. Use integers >= 1.")

    raster_histogram_with_mask(fn, mask_fn, band_select)

