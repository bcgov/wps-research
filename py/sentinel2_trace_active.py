'''demo method to map an effective fire boundary'''
import os
import sys
from misc import err, run, pd, sep, exists, args
cd = pd + '..' + sep + 'cpp' + sep

if len(args) < 2:
    err('sentinel2_trace_active.py [input file, binary mask envi type 4]')

fn = args[1]
if not exists(fn):
    err('please check input file')
    
for i in [50]: # [10, 20, 60]: # 90 
    run(cd + 'flood.exe ' + fn)
    run(cd + 'class_link.exe ' + fn + '_flood4.bin  ' + str(i)) # 40')
    run(cd + 'class_recode.exe ' + fn + '_flood4.bin_link.bin 1')
    run(cd + 'class_wheel.exe ' + fn + '_flood4.bin_link.bin_recode.bin')
    run('python3 ' + pd + 'raster_plot.py ' + fn + '_flood4.bin_link.bin_recode.bin_wheel.bin  1 2 3 1 1')
    lines = os.popen(cd + 'class_onehot.exe ' + fn + '_flood4.bin_link.bin_recode.bin').readlines()

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
                    continue
                N = int(f.split('.')[-2].split('_')[-1])
                print(N)

                lines = os.popen(cd + 'binary_hull.exe ' + f).readlines()
                for line in lines:
                    print(line.strip())

                run('python3 ' + pd + 'raster_plot.py ' + f + ' 1 1 1 1 1')
                run('mv alpha_shape.png alpha_shape_' + str(N) + '.png')
            except:
                pass
    # run('mv test.bin_flood4.bin_link.bin_recode.bin_wheel.bin_1_2_3_rgb.png test' + str(i) + '.png')
