'''link two tables approximately using distance comparison of lat/lon values

20220506 adapt this to arbitrary sheets

e.g. python3 ~/GitHub/wps-research/py/geo_join/join.py ../Feb-16th-2022-03-35PM-Flight-Airdata.csv_select-select_file-exclude-.csv latitude longitude exif_table.csv decimal_latitude decimal_longitude
'''
import os
import sys
args = sys.argv
exist = os.path.exists
if exist('merge.csv'):
    print('Error: merge.csv already exists'); sys.exit(1)

# default parameters used in ftl project before:
ftl_f = 'ftl.csv'
ftl_y = 'y'
ftl_x = 'x'
spc_f = 'spectra.csv'
spc_lat = 'ctr_lat'
spc_lon = 'ctr_lon'

if len(args) > 6:
    ftl_f, ftl_x, ftl_y, spc_f, spc_lat, spc_lon = args[1: 7]  # oops switched the ftl_x and ftl_y here!!! CHECK COORDINATES !!
else:
    print("Error: check args"); sys.exit(1)

def records(csv_fn):
    f = open(csv_fn)
    if not f:
        print("Err: failed to open file:", csv_fn)
    lines = f.readlines()
    hdr = lines[0].split(",")
    hdr = [x.strip().lower() for x in hdr]
    n = len(hdr)
    data = {x: [] for x in hdr}
    for i in range(1, len(lines)):
        w = lines[i].strip().split(',')
        if len(w) != n:
            print("Error:", w)
        w = [x.strip() for x in w]
        for j in range(0, n):
            data[hdr[j]].append(w[j])
    #print("*** keys", list(data.keys())[:10])
    return data

# ftl, spc = records("ftl.csv"), records("spectra.csv")
ftl, spc = records(ftl_f), records(spc_f) 
# cat spc record data onto ftl record data (nearest match)
#print("ftl", ftl)

# x, y = ftl['x'], ftl['y']
x, y = ftl[ftl_x], ftl[ftl_y]
#print("x", x)
#print("y", y)

#lat, lon = spc['ctr_lat'], spc['ctr_lon']
lat, lon = spc[spc_lat], spc[spc_lon]

print("LEN(LAT)", len(lat))
print("LEN(x)", len(x))
def flt(q):
    return [float(i) for i in q]

x, y, lat, lon = flt(x), flt(y), flt(lat), flt(lon)

f = open("merge.csv","wb")
f.write((','.join(list(ftl.keys()) + list(spc.keys()))).encode())

m_j, m_x, max_r = [], [], 0.
for i in range(0, len(lat)): 
    # print("i", i) 
    sx, sy = lat[i], lon[i] # for each spectra row, find ftl row ID that matches
    min_j, min_d = None, None
    for j in range(0, len(x)):
        d = abs(x[j] - sx) + abs(y[j] - sy)
        print("\tj", j, "\td", round(d,5), "\txj", round(x[j],5), "\tsx", round(sx,5), "\tyj", round(y[j],5), "sy", round(sy,5))
        if j == 0 or d < min_d:
            min_j, min_d = j, d # start with first compare
    print("i", i, "min_j", min_j, "min_d", min_d)
    m_j.append(min_j)  
    # print("len(lat)", len(lat), "lat", lat)
    m_x.append(lat[min_j])
    frc = [ftl[k][min_j] for k in ftl.keys()]
    src = [spc[k][i] for k in spc.keys()]
    f.write(('\n' + ','.join(frc + src)).encode())
    max_r = min_d if min_d > max_r else max_r
    # sys.exit(1)
f.close()
print("max distance:", max_r)
if len(list(set(m_j))) != len(m_j):
    print(len(list(set(m_j))), len(m_j))
    print("Error: not 1-1"); sys.exit(1)

N_m_j = len(list(set(m_j)))
print("N_match", N_m_j)
if N_m_j != len(x) and N_m_j != len(lon):
    print("ERROR") # check we matched the right number of records
else:
    print("SUCCESS")
print("max distance:", max_r)
print("smaller is better")
