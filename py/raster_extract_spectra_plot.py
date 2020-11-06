import os
import sys
import matplotlib.pyplot as plt
args = sys.argv

if len(args) < 2:
    print("Error: [input spectra file.csv]"); sys.exit(1)

lines = open(args[1]).readlines()
lines = [x.strip() for x in lines]

circles = []

ci, x_min, y_min = 0, None, None
x_max, y_max = None, None

X = []
Y = []
plt.figure()
for line in lines[1:]:
    w = line.split(",")
    w = w[4:]
    pix, lin = int(w[0]) + int(w[2]), -(int(w[1]) + int(w[3]))
    pix, lin = float(pix), float(lin)
    ci += 1 
    r, g, b, = w[4], w[5], w[6]
    r, g, b = float(r), float(g), float(b)
    [r, g, b] = [r/1650., g/1650., b/1650.]
    print("  ", pix, lin, [r,g,b])
    plt.scatter([pix], [lin], color=(r,g,b)) #, 'x')
    X.append(pix)
    Y.append(lin)

#  plt.plot(X,Y, 'x')
plt.show()
# fig.savefig('plotcircles.png', dpi = 3000)

