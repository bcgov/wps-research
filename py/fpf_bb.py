from misc import err, exists, run
from bounding_box import bounding_box

d = open("fpf").read().strip()
print(d)

if len(d.split('POLY')) > 1:
    err("input needs to be point coordinates (lat/lon) only")

w = d.split(',')
if len(w) != 2:
    err("Single point coordintes (lat/lon) only supported")

lat = float(w[0].split('(')[1].strip())
lon = float(w[1].strip(')').strip())

print(lat, lon)

bb = bounding_box(lat, lon, 10)

p = [[bb[0], bb[1]],
     [bb[0], bb[3]],
     [bb[2], bb[3]],
     [bb[2], bb[1]]]

p = [ str(x[1]) + " " + str(x[0]) for x in p]
s = "Intersects(POLYGON((" + (','.join(p)) + ")))"

print(s)

run("cp -v fpf fpf_bak")
print("+w fpf")
open("fpf", "wb").write(s.encode())
