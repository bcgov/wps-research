'''20230712 this version deprecated. Removed alpha shape functionality. 

Replaced with binary_polygonize.py

Original documentation:
demo method to map an effective fire boundary

n.b. would need to place results in separate folders for running in parallel (script needs to be cleaned up)'''
BRUSH_SIZE = 1111
POINT_THRES = 10 # don't get hulls for shapes with less than 50 points
WRITE_PNG = False # set to true for debug visuals
import os
import sys
from envi_header_copy_mapinfo import envi_header_copy_mapinfo
from misc import err, run, pd, sep, exists, args, hdr_fn
cd = pd + '..' + sep + 'cpp' + sep

if len(args) < 2:
    err('sentinel2_trace_active.py [input file, binary mask envi type 4] [optional arg: alpha shape parameter]') 

shape_parameter = None
class_select = None 
if len(args) > 2:
    shape_parameter = float(args[2])

fn = args[1]
if not exists(fn):
    err('please check input file')
hfn = hdr_fn(args[1])    


for i in [BRUSH_SIZE]: # [10, 20, 60]: # 90   # 150
    if not exists(fn + '_flood4.bin'): # first output file
        run(['ulimit -s 1000000;' + cd + 'flood.exe ' + fn])

        envi_header_copy_mapinfo(['envi_header_copy_mapinfo', hfn, fn + '_flood4.hdr'])


    if not exists(fn + '_flood4.bin_link.bin'):  # second output file
        run([cd + 'class_link.exe',
             fn + '_flood4.bin',
             str(i)]) # 40')


        envi_header_copy_mapinfo(['envi_header_copy_mapinfo', hfn, fn + '_flood4.bin_link.hdr'])


    if not exists(fn + '_flood4.bin_link.bin_recode.bin'):  # third output file
        run([cd + 'class_recode.exe',
             fn + '_flood4.bin_link.bin',
             '1'])

        envi_header_copy_mapinfo(['envi_header_copy_mapinfo', hfn, fn + '_flood4.bin_link.bin_recode.hdr'])


    if not exists(fn + '_flood4.bin_link.bin_recode.bin_wheel.bin'):
        run([cd + 'class_wheel.exe',
             fn + '_flood4.bin_link.bin_recode.bin'])

        run(['python3 ' + pd + 'raster_plot.py',
             fn + '_flood4.bin_link.bin_recode.bin_wheel.bin',
             '1 2 3 1 1'])
    
    cmd = cd + 'class_onehot.exe ' + fn + '_flood4.bin_link.bin_recode.bin ' + str(POINT_THRES)
    print(cmd)
    lines = os.popen(cmd).read().split('\n')
    for line in lines:
        print(line)
    '''
    lines = None
    run('rm -f class_onehot.dat')
    if not exists('class_onehot.dat'):
        lines = os.popen(cmd).read()
        open('class_onehot.dat', 'wb').write(lines.encode())
    else:
        lines = open('class_onehot.dat').read()
    '''
    for line in lines:
        w, f = line.strip().split(), None
        try:
            if w[1][-4:] == '.hdr' and w[0] == '+w':
                f = w[1][:-4] + '.bin'
                print(f)
        except:
            pass
        if f:
            if True:
                if f == fn + '_flood4.bin_link.bin_recode.bin_1.bin':
                    continue  # first class should be empty !
                N = int(f.split('.')[-2].split('_')[-1]) # print(N)
                if class_select is not None:
                    if N != class_select:
                        continue

                # point count this class: skip if under threshold
                print('run(' + cd + 'class_count.exe ' + f + ')')
                c = ''.join(os.popen(cd + 'class_count.exe ' + f).read().split())
                c = c.strip('{').strip('}').split(',')
                if len(c) != 2:
                    err('expected tuple of len 2')
                print("c", c)
                n_px = int(c[1].split(':')[1])
                if n_px < POINT_THRES:
                    print('********* SKIP this class,', n_px, ' below threshold: ', POINT_THRES) 
                    continue
          
                # create outline (alpha shape)
                run(['python3',
                    pd + 'alpha_shape.py',
                    f, "0.1" if shape_parameter is None else str(shape_parameter)])

                # plotting
                png_f = f + '_1_1_1_rgb.png'
                if not exists(f  + png_f) and WRITE_PNG:
                    run('python3 ' + pd + 'raster_plot.py ' + f + ' 1 1 1 1 1')

                # copy map info
                #run(['python3 ' + pd + 'envi_header_copy_mapinfo.py',
                #        fn[:-4] + '.hdr',
                #        f[:-4] + '.hdr'])

                envi_header_copy_mapinfo(["envi_header_copy_mapinfo", hfn, f[:-4] + '.hdr'])

                # generate KML
                ptfn = f + '_alpha_points.txt'
                run(['python3',
                     pd + 'raster_pixel_loc.py',
                     f,
                     ptfn]) # open(ptfn).read().strip()])
