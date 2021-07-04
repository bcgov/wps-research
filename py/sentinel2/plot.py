'''for a folder of zip files, make a plot of frames downloaded per day'''
import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

c = {}
lines = os.popen("ls -latrh --time-style=long-iso *.zip").readlines()
lines = [x.strip() for x in lines]
for line in lines:
    w = line.split()
    d = w[5]
    if d not in c:
        c[d] = 0
    c[d] += 1

X = []
Y = []
for d in c:
    x = d.split('-')
    x = [int(i) for i in x]  
    yy,mm,dd = x
    dt = datetime.datetime(yy,mm,dd)
    X.append(dt)
    Y.append(c[d])
X = X[:-1]
Y = Y[:-1]
plt.figure(figsize=(10,5))
plt.plot(X, Y, label='frames per day S2')
plt.plot(X, np.ones(len(Y)) * np.mean(np.array(Y)), label='average frames / day')
plt.ylim((0,None))
plt.legend()
plt.tight_layout()
plt.savefig("plot.png")
