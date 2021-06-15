import os
import sys
import pickle
import matplotlib.pyplot as plt
sep = os.path.sep

bins = os.popen("ls -1 brad/*.bin").readlines()
bins = [f.strip() for f in bins]

for b in bins:
    print("+r " + b)
    pfn = b.split(sep)[-1] + '.png'
    X = pickle.load(open(b, 'rb'))
    if not os.path.exists(pfn):
        plt.imshow(X)
        plt.tight_layout()
        print("+w " + pfn)
        plt.savefig(pfn)

'''
c = {}
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        d = X[i,j]
        if not d in c:
            c[d] = 0
        c[d] += 1
print(c)
'''
