#!/usr/bin/python3
'''20211202 simple version that was stripped down from 2020 ftl mvp version
    usage: e.g:
        python rasterplot.py sentinel2.bin 4 3 2
    
    tested on Python 3.8.10 (default, Sep 28 2021, 16:10:42)
    with numpy.version.version '1.20.2' and matplotlib.__version__ '3.4.1'
    
    e.g. installation of numpy and matplotlib (Ubuntu):
   sudo apt install python-matplotlib python-numpy


20241202 add flag to perform 1-d ( | r,g,b | ) scaling, instead of default: 
|r|, |g|, |b| ( separate, l2 ) scaling. In both cases, histogram stretching.
'''
import warnings; warnings.filterwarnings("ignore", message="Unable to import Axes3D")
from misc import *
import matplotlib

def naninf_list(x):
    Y = []
    X = None
    if type(x) != list:
        X = list(x.ravel().tolist())
    else:
        X = x

    for i in X:
        if not (math.isnan(i) or math.isinf(i)):
            Y.append(i)
    return Y

def nanmin(x):
    Y = naninf_list(x)
    xm = Y[0]
    for i in Y:
        if i < xm:
            xm = i;
    return xm

def nanmax(x):
    Y = naninf_list(x)
    xm = Y[0]
    for i in Y:
        if i > xm:
            xm = i;
    return xm

if __name__ == '__main__':

    args = sys.argv
    keys, args_new = [], []
    for arg in args:
        if arg[0:2] == '--':
            keys += [arg[2:]]
        else:
            args_new += [arg]
    args = args_new
    print("keys", keys)
    print("args", args)

    # instructions to run
    if len(args) < 2:
        err('usage:\n\tread_multispectral.py [input file name]' +
            ' [optional: red band idx]' +
            ' [optional: green band idx]' +
            ' [optional: blue band idx] #band idx from 1' + 
            ' [optional: background plotting]' + 
            ' [optional: no hist trimming]') 
            # + ' [optional: class_legend (not implemented)]')
    fn, hdr = sys.argv[1], hdr_fn(sys.argv[1])  # check header exists
    assert_exists(fn)  # check file exists
    
    skip_plot = True if (len(args) > 5  and args[5] == '1') else False
    use_trim = False if (len(args) > 6  and args[6] == '1') else True
    # class_legend = True if (len(args) > 7 and args[7] == '1') else False

    samples, lines, bands = read_hdr(hdr)  # read header and print parameters
    for f in ['samples', 'lines', 'bands']:
        exec('print("' + f + ' =" + str(' +  f + '))')
        exec(f + ' = int(' + f + ')')
    
    npx = lines * samples # number of pixels.. binary IEEE 32-bit float data
    data = read_float(sys.argv[1]).reshape((bands, npx))
    print("bytes read: " + str(data.size))
    
    bn = None
    try: bn = band_names(hdr)  # try to read band names from hdr
    except: pass
    
    # select bands for visualization # band_select = [3, 2, 1] if bands > 3 else [0, 1, 2]
    band_select, ofn = [0, 1, 2], None
    try:  # see if we can set the (r,g,b) encoding (band selection) from command args
        for i in range(0, 3):
            bs = int(args[i + 2]) - 1
            if bs < 0 or bs >= bands:
                err('band index out of range')
            band_select[i] = bs
    except: pass
    
    middle = None  # reproducibility: put band idx used, in output fn
    try: middle = args[2: 2 + 3]
    except: middle = [str(b + 1) for b in band_select]
    ofn = '_'.join([fn] + middle + ["rgb.png"])
    
    n_points, rgb = 0, np.zeros((lines, samples, 3))
    band_select = [0, 0, 0,] if bands == 1 else band_select # could be class map. or just one band map
    bn = [bn[i] for i in band_select] if bn else bn  # cut out the band names used, if applicable
    print("band_select", band_select)
    
    def scale_rgb(i):  # for i in range(3)
        rfn = fn + '_rgb_scaling_' + str(i) + '.txt'
        rgb_min, rgb_max = None, None
        rgb_i = data[band_select[i], :].reshape((lines, samples))
    
        if use_trim: # if not override_scaling
            if not exists(rfn):
                values = rgb_i  # now do the so called x-% linear stretch (separate bands version)
                values = naninf_list(values) # values.reshape(np.prod(values.shape)).tolist()
                values.sort()
    
                if values[-1] < values[0]:   # sanity check
                    err("failed to sort")
    
                for j in range(0, len(values) -1): #npx - 1):
                    if values[j] > values[j + 1]:
                        err("failed to sort")
    
                n_pct = 1.5 # percent for stretch value
                frac = n_pct / 100.
                rgb_min, rgb_max = values[int(math.floor(float(len(values))*frac))],\
                               values[int(math.floor(float(len(values))*(1. - frac)))]
                print('+w', rfn)
                open(rfn, 'wb').write((','.join([str(x) for x in [rgb_min, rgb_max]])).encode())
                # DONT FORGET TO WRITE THE FILE HERE
            else:  # assume we can restore
                rgb_min, rgb_max = [float(x) \
                        for x in open(rfn).read().strip().split(',')]
                print('+r', rfn)
        else:
            rgb_min, rgb_max = nanmin(rgb_i), nanmax(rgb_i)
        print("i, min, max", i, rgb_min, rgb_max)
        rng = rgb_max - rgb_min  # apply restored or derived scaling
        rgb_i = (rgb_i - rgb_min) / (rng if rng != 0. else 1.)
        rgb_i[rgb_i < 0.] = 0.  # clip
        rgb_i[rgb_i > 1.] = 1.
        return rgb_i
    
    def scale_rgb_global():  # scale |r| + |g| + |b| (l2) instead of |r|, |g|, |b| separately
        print("scale_rgb_global..")
        rfn = fn + '_rgb_scaling.txt'
        rgb_min, rgb_max = None, None
        
        '''
        values = []
        for j in range(lines * samples):
            my_values = [data[band_select[k], j] for k in range(3)]
            values += [max(max(my_values[0], my_values[1]), my_values[2])]
        '''

        if use_trim: # if not override_scaling
            if not exists(rfn):

                values = []
                for j in range(lines * samples):
                    my_values = [data[band_select[k], j] for k in range(3)]
                    values += [max(max(my_values[0], my_values[1]), my_values[2])]

                values = naninf_list(values) # values.reshape(np.prod(values.shape)).tolist()
                values.sort()

                if values[-1] < values[0]:   # sanity check
                    err("failed to sort")

                for j in range(0, len(values) -1): #npx - 1):
                    if values[j] > values[j + 1]:
                        err("failed to sort")

                n_pct = 1. # percent for stretch value
                frac = n_pct / 100.
                rgb_min, rgb_max = values[int(math.floor(float(len(values))*frac))],\
                               values[int(math.floor(float(len(values))*(1. - frac)))]
                print('+w', rfn)
                open(rfn, 'wb').write((','.join([str(x) for x in [rgb_min, rgb_max]])).encode())
                # DONT FORGET TO WRITE THE FILE HERE
            else:  # assume we can restore
                rgb_min, rgb_max = [float(x) \
                        for x in open(rfn).read().strip().split(',')]
                print('+r', rfn)
        else:
            rgb_min, rgb_max = nanmin(rgb_i), nanmax(rgb_i)

        for i in range(3):
            rgb_i = data[band_select[i], :].reshape((lines, samples))
            print("i, min, max", i, rgb_min, rgb_max)
            rng = rgb_max - rgb_min  # apply restored or derived scaling
            rgb_i = (rgb_i - rgb_min) / (rng if rng != 0. else 1.)
            rgb_i[rgb_i < 0.] = 0.  # clip
            rgb_i[rgb_i > 1.] = 1.
            rgb[:, :, i] = rgb_i


    if not 'global' in keys:
        use_parfor = True # False
        if use_parfor:
            rgb_i = parfor(scale_rgb, range(3), 3)
        else:
            rgb_i = [scale_rgb(i) for i in range(3)]
        for i in range(3):
            rgb[:, :, i] = rgb_i[i]
    else:
        scale_rgb_global()
        
    
    if True:  # plot image: no class labels
        if skip_plot:
            mpl = matplotlib
            COLOR = 'white'
            mpl.rcParams['text.color'] = COLOR
            mpl.rcParams['axes.labelcolor'] = COLOR
            mpl.rcParams['xtick.color'] = COLOR
            mpl.rcParams['ytick.color'] = COLOR
    
        base_in = 20.
        fig = plt.figure(frameon=True,
                         figsize=(base_in, base_in * float(lines) / float(samples)))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_frame_on(True)
        plt.axis('on')
        ff = os.path.sep.join((os.path.abspath(fn).split(os.path.sep))[:-1]) + os.path.sep
        title_s = fn.split("/")[-1] if not exists(ff + 'title_string.txt') else open(ff + 'title_string.txt').read().strip()
        x_label = ''
        bn = [(' ' + bn[i] if i > 0 else bn[i])  for i in range(len(bn))]
        if bn:
            x_label += '(R,G,B) = (' + (','.join(bn)) + ')'
        plt.title(title_s, fontsize=30)
        plt.style.use('dark_background')
        
        d_min, d_max = nanmin(rgb), nanmax(rgb)
        print("d_min", d_min, "d_max", d_max)
        # rgb = rgb / (d_max - d_min)
        rgb = np.nan_to_num(rgb, nan=0.0)
        plt.imshow(rgb) #, vmin = 0., vmax = 1.) #plt.tight_layout()
        print(ff)
        if exists(ff + 'copyright_string.txt'):
            x_label += (' ©' + open(ff+ 'copyright_string.txt').read().strip())
        plt.xlabel(x_label, fontsize=20)
        print("+w", ofn)
        plt.tight_layout()
        if not skip_plot:
            plt.show()
        fig.savefig(ofn, transparent=(not skip_plot))
