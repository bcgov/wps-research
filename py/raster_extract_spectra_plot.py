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
C = []
plt.figure()
for line in lines[1:]:
    w = line.split(",")
    w = w[4:]
    pix, lin = int(w[0]) + int(w[2]), -(int(w[1]) + int(w[3]))
    pix, lin = float(pix), float(lin)
    ci += 1 
    r, g, b, = w[4], w[5], w[6]
    r, g, b = float(r), float(g), float(b)
    X.append(pix)
    Y.append(lin)
    C.append( [r,g,b] )


C_min = list(C[0])
C_max = list(C[0])
for i in range(0, len(X)):
    for j in range(0, 3):
        if C[i][j] < C_min[j]: C_min[j] = C[i][j]
        if C[i][j] > C_max[j]: C_max[j] = C[i][j]

print("C_min", C_min)
print("C_max", C_max)

for i in range(0, len(X)):
    Ci = list(C[i])

    for j in range(0, 3):
        Ci[j] -= C_min[j]
        Ci[j] /= (C_max[j] - C_min[j])

    C[i] = Ci

for i in range(0, len(X)):
    r,g,b =  C[i]
    print(C[i])
    plt.scatter(X[i], Y[i], color=(r,g,b))
#  plt.plot(X,Y, 'x')
plt.show()
# fig.savefig('plotcircles.png', dpi = 3000)

