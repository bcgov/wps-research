'''20230712: trace multiple boundaries from one detection result
- match to bc fire point locations for identification.

hint: scrub the class map before applying this..

demo method to map an effective fire boundary

n.b. might place results in separate folders for running in parallel (script needs to be cleaned up)'''
BRUSH_SIZE = 50
POINT_THRES = 10 # don't get hulls for shapes with less than 50 points
WRITE_PNG = False # set to true for debug visuals
import os
import sys
from misc import err, run, pd, sep, exists, args
cd = pd + '..' + sep + 'cpp' + sep

if len(args) < 3:
    err('sentinel2_trace_active.py [input file, binary mask envi type 4] [source data file]')

fn, src_data = args[1], args[2]
if not exists(fn) or not exists(src_data):
    err('please check input file')
run('rm -v *crop*')

for i in [BRUSH_SIZE]: # [10, 20, 60]: # 90   # 150
    if not exists(fn + '_flood4.bin'):
        run(['ulimit -s 1000000;' + cd + 'flood.exe ' + fn])

    if not exists(fn + '_flood4.bin_link.bin'):
        run([cd + 'class_link.exe',
             fn + '_flood4.bin',
             str(i)]) # 40')

    if not exists(fn + '_flood4.bin_link.bin_recode.bin'):
        run([cd + 'class_recode.exe',
             fn + '_flood4.bin_link.bin',
             '1'])

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
    
    for line in lines:
        w, f, hfn = line.strip().split(), None, None
        try:
            if w[1][-4:] == '.hdr' and w[0] == '+w':
                f = w[1][:-4] + '.bin'
                print(f)
                hfn = w[1][:-4] + '.hdr'
        except:
            pass
        if f:
            if True:
                if f == fn + '_flood4.bin_link.bin_recode.bin_1.bin':
                    continue  # first class should be empty !
                
                run('cp ' + fn[:-4] + '.hdr ' + hfn)

                N = int(f.split('.')[-2].split('_')[-1]) # print(N)

                run('crop ' + f)
                run('pad ' + f + '_crop.bin ' + str(BRUSH_SIZE))
                f_0 = f
                f = f_0 + '_crop.bin_pad.bin'
                hfn = f_0 + '_crop.bin_pad.hdr'

                print('run(' + cd + 'class_count.exe ' + f + ')')  # skip if under threshold
                c = ''.join(os.popen(cd + 'class_count.exe ' + f).read().split())
                c = c.strip('{').strip('}').split(',')
                if len(c) != 2:
                    err('expected tuple of len 2')
                print("c", c)
                n_px = int(c[1].split(':')[1])
                
                if n_px < POINT_THRES:
                    print('********* SKIP this class,', n_px, ' below threshold: ', POINT_THRES) 
                    continue
          
                run(['python3',
                    pd + 'binary_polygonize.py',
                    f])

                f_i = str(N).zfill(3)
                src_clip = f_i + '.bin'
                src_cliph = f_i + '.hdr'    

                # project source data onto clip area
                run('po ' + src_data + ' ' + f + ' ' + src_clip)

                # convert shapefile and raster to same projection
                # run('


                # project fire locations onto clip area : COULDN'T GET THIS WORKING YET
                # cmd = 'python3 ~/GitHub/wps-research/py/shapefile_nearest_point_to_raster.py  ~/GitHub/wps-research/py/prot_current_fire_points.shp ' + src_clip
                #print(cmd)
                #lines = [x.strip() for x in os.popen(cmd).readlines()]

                #FIRE_NUM = None
                #for line in lines:
                #    w = line.split(':')
                #    if w[0] == "FIRE_NUM":
                #        FIRE_NUM = w[1]
                #  
                # print(FIRE_NUM)               
                FIRE_NUM = f_i


                # S2B_MSIL1C_20230712T191919_N0509_R099_T09UYV_20230712T212909.bin

                w = src_data.split('_')
                ts = w[2].split('T')
                ds = ts[0]
                ts = ts[1]
            
                hh = int(ts[0:2]) - 7  # convert utc to PST
                mm = int(ts[2:4]) 
    
                print(ts, hh, mm)

                string = '23_' + FIRE_NUM + "_" + ds + '_' + str(hh).zfill(2)  + str(mm).zfill(2) + '_detection_sentinel2'
                print(string)

                run('cp ' + f + '.kml ' + string + '.kml')
                run('mv ' + src_clip + ' ' + string + '.bin')
                run('mv ' + src_cliph + ' ' + string + '.hdr')

                binfile = string + '.bin'
                run('envi2tif.py ' + binfile)
                run('mv ' + binfile + '_ht.bin_smult.tif ' + string + '.tif')
run('chmod 755 23_*.tif')
run('chmod 755 23_*.kml')
run('rm *smult*')
run('rm *bin_ht*')
run('clean')
