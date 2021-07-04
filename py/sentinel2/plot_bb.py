'''plot bounding box counts for zip files data

NB need to vectorize the plot operation...'''
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

pickle.dump('X', open('.x', 'wb'))
x = pickle.load(open('.x', 'rb'))
if x != 'X':
    print("Error.")
    sys.exit(1)

c = None
if not os.path.exists('c.pkl'):
    ci = 0
    c = {}
    zf = [z.strip() for z in os.popen("ls -1 *.zip").readlines()]
    for z in zf:
        ci += 1
        if ci % 10 == 0:
            print(ci * 100. / len(zf))
        try:
            # print(z)
            dat = os.popen("gdalinfo " + z).read().strip().split('\n')
            
            poly = None
            for i in range(len(dat)):
                w = dat[i].split('=')
                if w[0].strip() == 'FOOTPRINT':
                    poly = dat[i]
            # print(poly)
            '''FOOTPRINT=POLYGON((-124.31644818066124 50.45577257376828, -124.28292003409229 50.53758463407535, -124.22726381956744 50.68419967922297, -124.16387059669955 50.828862837596304, -124.10622483971392 50.97495334742904, -124.04813902627198 51.12104328564767, -123.98887657354973 51.26691083616258, -123.92725580502349 51.412324197858325, -123.91374744532915 51.44525769803033, -123.08820053753058 51.4498312923452, -122.85965081326997 51.41463577226112, -122.86248750524196 50.463717167775926, -124.31644818066124 50.45577257376828))'''
            s = poly.strip()       
            # print(s)
            if not s in c:
                c[s] = 0
            c[s] += 1
        except:
            pass
    pickle.dump(c, open('c.pkl', 'wb'))
else:
    print("loading..")
    c = pickle.load(open('c.pkl', 'rb'))
    print("loaded..")

fig = plt.figure()
ax = plt.axes(projection='3d')
for s in c:
    count = c[s]
    s = s.split('((')[1].strip(')').strip(')')
    s = s.split(',')
    for x in s:
        x = x.split()
        w = [float(i) for i in x]
        ax.scatter(w[0], w[1], count) # need to put this stuff all in one vector!

plt.show() # savefig("plot_bb.png")

print("+w plot_bb.png")
