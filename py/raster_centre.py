'''20230530 find centre point of raster.
based on code from: 
https://en.proft.me/2015/09/20/converting-latitude-and-longitude-decimal-values-p/

** If more than one argument is supplied, calculate the average centre
'''
import sys
import os
import re
def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    if direction == 'S' or direction == 'W':
        dd *= -1
    return dd;

def dd2dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]

def parse_dms(dms):
    parts = re.split('[^\d\w]+', dms)
    parts[2] = parts[2].replace('_', '.')
    parts[6] = parts[6].replace('_', '.')
    lat = dms2dd(parts[0], parts[1], parts[2], parts[3])
    lng = dms2dd(parts[4], parts[5], parts[6], parts[7])

    return (lat, lng)

dd = parse_dms("36°57'9' N 110°4'21' W")

if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
    print("Usage:\n\traster_centre [raster file that GDAL can read]")
    sys.exit(1)


total = None
for i in range(1, len(sys.argv)):
    lines = [x.strip() for x in os.popen('gdalinfo ' + sys.argv[i]).readlines()]

    for line in lines:
        w = line.split()
        if w[0] == 'Center':
            w = ' '.join(line.split('(')[-1].strip(')').split(',')).replace('"', "'").replace('d', '°').replace('N', ' N').replace('E', ' E').replace('S', ' S').replace('W', ' W').replace('.', '_') 
            dd = parse_dms(w)
            print(dd)

            if total == None:
                total = [0,0]
                total[0], total[1] = dd[0], dd[1]
            else:
                total[0] += dd[0]
                total[1] += dd[1]
        
total =[ x/(len(sys.argv)-1) for x in total]
print("average", total)
