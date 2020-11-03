import os
import sys
import matplotlib.pyplot as plt
args = sys.argv

if len(args) < 2:
    print("Error: [input spectra file.csv]"); sys.exit(1)

lines = open(args[1]).readlines()
lines = [x.strip() for x in lines]

circles = []

fig, ax = plt.subplots()

ci, x_min, y_min = 0, None, None
x_max, y_max = None, None
for line in lines[1:]:
    w = line.split(",")
    w = w[4:]
    print(w)
    pix, lin = int(w[0]), -int(w[1])
    if ci == 0:
        x_min = x_max = pix
        y_min = y_max = lin
    else:
        x_min = pix if pix < x_min else x_min
        x_max = pix if pix > x_max else x_max
        y_min = lin if lin < y_min else y_min
        y_max = lin if lin > y_max else y_max 
    ci += 1 

    r, g, b, = w[4], w[5], w[6]
    r, g, b = float(r), float(g), float(b)
    [r, g, b] = [r/255., g/255., b/255.]
    c = plt.Circle((pix, lin), 150, color=(r,g,b))
    circles.append(c)


ax.set_aspect('equal')
for c in circles:
    ax.add_artist(c)

plt.xlim((x_min, x_max))
plt.ylim((y_min, y_max))
plt.xlim((0, 10980))
plt.ylim((-10980, 0))

fig.savefig('plotcircles.png', dpi = 1000)

