'''20240210: modified from raster_histogram.py 
20221105 update: add NAN resiliency (i.e. skip NAN valued pixels in stats) '''
from misc import *
import matplotlib
args = sys.argv
n_bins = 5000
if __name__ == '__main__':     # instructions to run
    if len(args) < 2:
        err('usage:\n\traster_histogram.py [input file name] [class map] [class index]')
    fn, hdr = sys.argv[1], hdr_fn(sys.argv[1])  # check header exists
    assert_exists(fn)  # check file exists

    # subselect data based on matching value
    fn2, hdr2 = sys.argv[2], hdr_fn(sys.argv[2])
    assert_exists(fn2)

    skip_plot = False
    samples, lines, bands = read_hdr(hdr)  # read header and print parameters
    for f in ['samples', 'lines', 'bands']:
        exec('print("' + f + ' =" + str(' +  f + '))')
        exec(f + ' = int(' + f + ')')

    samples2, lines2, bands2 = read_hdr(hdr2)
    [samples, lines, bands, samples2, lines2, bands2] = [int(x) for x in [samples, lines, bands, samples2, lines2, bands2]]
    if samples2 != samples or lines2 != lines or bands2 != 1:
        print(samples, lines, bands)
        print(samples2, lines2, bands2)
        err('unexpected dimensions on class map')
    
    npx = lines * samples # number of pixels.. binary IEEE 32-bit float data
    data = read_float(sys.argv[1]).reshape((bands, npx))
    print("bytes read: " + str(data.size))
    
    dat2 = read_float(sys.argv[2])
    ix =  (dat2 == float(sys.argv[3]))
    

    bn = None
    try: bn = band_names(hdr)  # try to read band names from hdr
    except: pass
    
    band_select = [i for i in range(bands)]
    N = len(band_select)
    rng = range(N)
    dat = [data[band_select[i],] for i in rng]
    my_min = [np.nanmin(dat[i][ix]) for i in rng]
    my_max = [np.nanmax(dat[i][ix]) for i in rng]
    print("min", my_min)
    print("max", my_max)
    my_min = np.nanmin(my_min)
    my_max = np.nanmax(my_max)
    print("min", my_min, "max", my_max)
    bs = (my_max - my_min) / n_bins
    bins = [my_min + float(i + 1) *bs for i in range(n_bins)]
    print(bins)
    print("max/bs", math.floor((my_max - my_min) / bs))
    print("min/bs", math.floor(((bs * 1.) + my_min - my_min) / bs))
    
    M = len(dat[0][ix])

    plt.figure()
    plt.title(sys.argv[1] + ' with ' + sys.argv[2] + '=' + str(sys.argv[3]))
    for i in range(N):
        di = dat[i][ix]
        plt.hist(di, range=[my_min, my_max], bins=n_bins, histtype='step', label=bn[band_select[i]])
    plt.tight_layout()
    plt.legend()
    plt.show()

