'''Raster histogram: by default (no parameters) all bands get their own color histogram,
together on the same plot. 

If we are dealing with RGB or 3-channel representations, we could select 
three bands only in which case histograms are plotted for these three
bands only (red, green, blue respectively)

20221105 update: add NAN resiliency (i.e. skip NAN valued pixels in stats) '''
from misc import *
import matplotlib
args = sys.argv
n_bins = 5000

'''select input file and select the three indices to be used for the false-color representation'''
def raster_histogram(input_file, band_select = [0, 1, 2]):
    assert_exists(input_file)  # check file exists
    hdr = hdr_fn(input_file)  # locate header file
    skip_plot = False
    [samples, lines, bands] = [int(x) for x in read_hdr(hdr)]  # read header and print parameters
    
    npx = lines * samples # number of pixels.. binary IEEE 32-bit float data
    data = read_float(input_file).reshape((bands, npx))
    # print("bytes read: " + str(data.size))

    bn = None
    try:
        bn = band_names(hdr)  # try to read band names from hdr
    except:
        pass

    N = len(band_select)  
    rng = range(N)
    dat = [data[band_select[i],] for i in rng]

    my_min, my_max = [np.nanmin(dat[i]) for i in rng],\
                     [np.nanmax(dat[i]) for i in rng]

    my_min, my_max  = np.nanmin(my_min),\
                      np.nanmax(my_max)
    
    bs = (my_max - my_min) / n_bins
    bins = [my_min + float(i + 1) * bs for i in range(n_bins)]
    M = len(dat[0])
    
    plt.figure()
    col = ['r', 'g', 'b']
    for i in range(N):
        plt.hist(dat[i],
                 range=[my_min, my_max],
                 bins=n_bins,
                 histtype='step',
                 label=bn[band_select[i]],
                 color=(col[i] if i < 3 else None))
    plt.tight_layout()
    plt.legend()
    plt.show()


def raster_transect(input_file, band_select = [0, 1, 2], row_index= None):
    assert_exists(input_file)  # check file exists
    hdr = hdr_fn(input_file)  # locate header file
    skip_plot = False
    [samples, lines, bands] = [int(x) for x in read_hdr(hdr)]  # read header and print parameters
    print(samples, lines, bands)

    npx = lines * samples # number of pixels.. binary IEEE 32-bit float data
    data = read_float(input_file).reshape((bands, npx))
    # print("bytes read: " + str(data.size))

    bn = None
    try:
        bn = band_names(hdr)  # try to read band names from hdr
    except:
        pass

    N = len(band_select)
    rng = range(N)
    dat = [data[band_select[i],] for i in rng]

    print(dat[0].shape)

    plt.figure()
    col = ['r', 'g', 'b']
    for i in range(N):
        plt.plot(range(samples),
                 dat[i][samples * row_index: samples * (row_index + 1)],
                 label = bn[band_select[i]],
                 color = (col[i] if i < 3 else None))
    plt.tight_layout()
    plt.legend()
    plt.legend()
    plt.show()




if __name__ == "__main__":
    if len(args) < 2:
        err('usage:\n\traster_histogram.py [input file name]' +
            ' [optional: red band idx]' +
            ' [optional: green band idx]' +
            ' [optional: blue band idx] #band idx from 1')
    
    fn, hdr = sys.argv[1], hdr_fn(sys.argv[1])  # check header exists

    band_select = [int(x) for x in [sys.argv[2], sys.argv[3], sys.argv[4]]]
    # make it go
    raster_histogram(fn, band_select)
