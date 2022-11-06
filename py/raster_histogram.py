'''20221105 update: add NAN resiliency (i.e. skip NAN valued pixels in stats) '''
from misc import *
import matplotlib
args = sys.argv

n_bins = 5000

if __name__ == '__main__':     # instructions to run
    if len(args) < 2:
        err('usage:\n\traster_histogram.py [input file name]' +
            ' [optional: red band idx]' +
            ' [optional: green band idx]' +
            ' [optional: blue band idx] #band idx from 1')
    fn, hdr = sys.argv[1], hdr_fn(sys.argv[1])  # check header exists
    assert_exists(fn)  # check file exists

    skip_plot = False
    # if len(args) > 5:
    #    skip_plot = True

    samples, lines, bands = read_hdr(hdr)  # read header and print parameters
    for f in ['samples', 'lines', 'bands']:
        exec('print("' + f + ' =" + str(' +  f + '))')
        exec(f + ' = int(' + f + ')')

    npx = lines * samples # number of pixels.. binary IEEE 32-bit float data
    data = read_float(sys.argv[1]).reshape((bands, npx))
    print("bytes read: " + str(data.size))

    bn = None
    try:
        bn = band_names(hdr)  # try to read band names from hdr
    except:
        pass

    # select bands for visualization # band_select = [3, 2, 1] if bands > 3 else [0, 1, 2]
    band_select, ofn = [0, 1, 2], None
    if len(args) < 5:
        band_select = [i for i in range(bands)]
    else:
        try:  # see if we can set the (r,g,b) encoding (band selection) from command args
            for i in range(0, 3):
                bs = int(args[i + 2]) - 1
                if bs < 0 or bs >= bands:
                    err('band index out of range')
                band_select[i] = bs
        except:
            pass
    
    N = len(band_select)
    rng = range(N)
    dat = [data[band_select[i],] for i in rng]
    my_min = [np.nanmin(dat[i]) for i in rng]
    my_max = [np.nanmax(dat[i]) for i in rng]
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
    
    M = len(dat[0])

    plt.figure()

    for i in range(N):
        di = dat[i]
        plt.hist(di, range=[my_min, my_max], bins=n_bins, histtype='step', label=bn[band_select[i]])
    plt.tight_layout()
    plt.legend()
    plt.show()

