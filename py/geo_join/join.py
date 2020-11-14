# link two tables approximately using distance comparison of lat/lon values
import os
import sys

def records(csv_fn):
    lines = open(csv_fn).readlines()
    hdr = lines[0].split(",")
    hdr = [x.strip().lower() for x in hdr]
    n = len(hdr)
    data = {x: [] for x in hdr}
    for i in range(1, len(lines)):
        w = lines[i].strip().split(',')
        if len(w) != n: print("Error:", w)
        w = [x.strip() for x in w]
        for j in range(0, n): data[hdr[j]].append(w[j])
    return data

ftl, spc = records("ftl.csv"), records("spectra.csv")
# cat spc record data onto ftl record data (nearest match)

x, y = ftl['x'], ftl['y']
lat, lon = spc['ctr_lat'], spc['ctr_lon']
def flt(q): return [float(i) for i in q]

x, y, lat, lon = flt(x), flt(y), flt(lat), flt(lon)

f = open("merge.csv","wb")
f.write((','.join(list(ftl.keys()) + list(spc.keys()))).encode())

m_j, m_x, max_r = [], [], 0.
for i in range(0, len(lat)): 
    sx, sy = lat[i], lon[i] # for each spectra row, find ftl row ID that matches
    min_j, min_d = None, None
    for j in range(0, len(x)):
        d = abs(x[j] - sx) + abs(y[j] - sy)
        if j == 0 or d < min_d: min_j, min_d = j, d # start with first compare
    print(i, min_j, min_d)
    m_j.append(min_j)  
    m_x.append(lat[min_j])
    frc = [ftl[k][min_j] for k in ftl.keys()]
    src = [spc[k][i] for k in spc.keys()]
    f.write(('\n' + ','.join(frc + src)).encode())
    max_r = min_d if min_d > max_r else max_r
f.close()

if len(list(set(m_j))) != 19:
    print("ERROR") # check we matched the right number of records
else:
    print("SUCCESS")
print("max distance:", max_r)
print("smaller is better")
