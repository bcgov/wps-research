skip = True

import os
import sys
sep = os.path.sep
exists = os.path.exists
last_step = '07_Box.data'  # local directory for last processing step

lines = [x.strip() for x in os.popen("ls -1").readlines()]

is_quad_pol = True

by_date = []
for line in lines:
    if len(line.split('ALOS')) > 1:
        if os.path.isdir(line):
            f = line
            date = '20' + line.split('-')[-1]
            print(f, date)
            dd = line + sep + last_step
            
            c11 = dd + sep + 'C11.bin'
            t11 = dd + sep + 'T11.bin'
            df = t11
            
            if exists(c11):
                is_quad_pol = False
                df = c11

            by_date += [[date, f, dd, df]] 
            
            # a = os.system("find ./" + line)

if len(by_date) < 2:
    print("Error: only one date")
    sys.exit(1)

by_date.sort()
for d in by_date:
    print(d)

if not exists(by_date[0][0]):
    os.mkdir(by_date[0][0])
cmd = ('cp -v ' + by_date[0][2] + sep + '* ' + by_date[0][0] + sep)
print(cmd)
if not skip:
    a = os.system(cmd) # put this back in at the end

pngs = []
for i in range(1, len(by_date)):
    print('\t',by_date[i])

    if not exists(by_date[i][0]):
        os.mkdir(by_date[i][0])
    
    cmd = 'raster_project_onto_all.py ' + by_date[i][2] + ' ' + by_date[0][3] + ' ' + by_date[i][0] + sep
    print(cmd)

    if not skip:
        a = os.system(cmd)

# regenerate png files:
for i in range(0, len(by_date)):
    d = by_date[i][0]
    p_d = by_date[i][1] + '_rgb.bin'
    if not exists(p_d) and not skip:
        if is_quad_pol:
            a = os.system('cd ' + d + sep + '; raster_stack.py T22.bin T33.bin T11.bin ' + by_date[i][1] + '_rgb.bin; raster_zero_to_nan ' + d + '_rgb.bin') 
        elif not(is_quad_pol):
            a = os.system('cd ' + d + sep + '; convert_iq_to_cplx C12_real.bin C12_imag.bin C12.bin; abs C12.bin;  raster_stack.py C11.bin C22.bin C12.bin_abs.bin ' + by_date[i][1] + '_rgb.bin; raster_zero_to_nan ' + by_date[i][1] + '_rgb.bin')
        else:
            err('unexpected mode')

    p_10 = d + sep + by_date[i][1] + '_rgb.bin_1_2_3_rgb.png'
    if not exists(p_10) and not skip:
        a = os.system('cd ' + d + sep + '; raster_plot.py ' + by_date[i][1] + '_rgb.bin 1 2 3 1')

    pngs += [p_10]

cmd = 'convert -delay 111 ' + ' '.join(pngs) + ' sequence.gif'
print(cmd)
if not exists('sequence.gif'):
    a = os.system(cmd)

for p in pngs:
    d = p.split(sep)[0]
    cmd = 'cp ' + p + ' ' + d + '.png'
    print(cmd)
    if not exists(d + '.png'):
        a = os.system(cmd)
