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

# circle1 = plt.Circle((0, 0), 0.2, color='r')
# circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
# circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)

# fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()


for c in circles:
    ax.add_artist(c)

#ax.add_artist(circle1)
#ax.add_artist(circle2)
#ax.add_artist(circle3)
plt.xlim((x_min, x_max))
plt.ylim((y_min, y_max))
plt.xlim((0, 10980))
plt.ylim((-10980, 0))

fig.savefig('plotcircles.png')

