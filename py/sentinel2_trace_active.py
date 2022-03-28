'''demo method to map an effective fire boundary


n.b. would need to place results in separate folders for running in parallel (script needs to be cleaned up)'''
POINT_THRES = 50 # don't get hulls for shapes with less than 50 points
import os
import sys
from misc import err, run, pd, sep, exists, args
cd = pd + '..' + sep + 'cpp' + sep

if len(args) < 2:
    err('sentinel2_trace_active.py [input file, binary mask envi type 4] [optional arg: class index]')

class_select = None 
if len(args) > 2:
    class_select = int(args[2])
print(class_select)

fn = args[1]
if not exists(fn):
    err('please check input file')
    
for i in [150]: # [10, 20, 60]: # 90 
    if not exists(fn + '_flood4.bin'):
        run('ulimit -s 1000000')
        run(cd + 'flood.exe ' + fn)
    if not exists(fn + '_flood4.bin_link.bin'):
        run(cd + 'class_link.exe ' + fn + '_flood4.bin  ' + str(i)) # 40')
    if not exists(fn + '_flood4.bin_link.bin_recode.bin'):
        run(cd + 'class_recode.exe ' + fn + '_flood4.bin_link.bin 1')
    if not exists(fn + '_flood4.bin_link.bin_recode.bin_wheel.bin'):
        run(cd + 'class_wheel.exe ' + fn + '_flood4.bin_link.bin_recode.bin')
        run('python3 ' + pd + 'raster_plot.py ' + fn + '_flood4.bin_link.bin_recode.bin_wheel.bin  1 2 3 1 1')
    print('class onehot..')
    cmd = cd + 'class_onehot.exe ' + fn + '_flood4.bin_link.bin_recode.bin ' + str(POINT_THRES)
    print(cmd)
    lines = None
    if not exists('class_onehot.dat'):
        lines = os.popen(cmd).read()
        open('class_onehot.dat', 'wb').write(lines.encode())
    else:
        lines = open('class_onehot.dat').read()

    lines = lines.split('\n')

    for line in lines:
        print(line)
    for line in lines:
        w, f = line.strip().split(), None
        try:
            if w[1][-4:] == '.hdr' and w[0] == '+w':
                f = w[1][:-4] + '.bin'
                print(f)
        except:
            pass
        if f:
            try:
                if f == fn + '_flood4.bin_link.bin_recode.bin_1.bin':
                    continue  # first class should be empty !
                N = int(f.split('.')[-2].split('_')[-1])
                print(N)
                if class_select is not None:
                    if N != class_select:
                        continue

                c = ''.join(os.popen(cd + 'class_count.exe ' + f).read().split())
                c = c.strip('{').strip('}').split(',')
                if len(c) != 2:
                    err('expected tuple of len 2')
                print("c", c)
                n_px = int(c[1].split(':')[1])
                if n_px < POINT_THRES:
                    print('********* SKIP this class,', n_px, ' below threshold: ', POINT_THRES) 
                    continue

                apfn = f + '_alpha_shape.pkl'
                if not exists(apfn):
                    lines = os.popen(cd + 'binary_hull.exe ' + f).readlines()
                    run('rm qhull.dat')
                    run('rm alpha_shape_input_file.txt')
                    # run('mv alpha_shape.png alpha_shape_' + str(N) + '.png')
                    # run('mv alpha_shape.pkl alpha_shape_' + str(N) + '.pkl')
                    # run('mv alpha_points.txt alpha_points_' + str(N) + '.txt')
                for line in lines:
                    print(line.strip())
                png_f = f + '_1_1_1_rgb.png'
                if not exists(f  + png_f):
                    run('python3 ' + pd + 'raster_plot.py ' + f + ' 1 1 1 1 1')
                # run('mv alpha_shape.png alpha_shape_' + str(N) + '.png')
                # run('mv alpha_shape.pkl alpha_shape_' + str(N) + '.pkl')

                # print out the coords here and pass to KML generator! 
            except:
                pass
    rm('class_onehot.dat')
