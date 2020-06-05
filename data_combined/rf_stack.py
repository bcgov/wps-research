# n_skip: use every n_skip-th pixel for trainingral gun ban quietly expanded, potentially putting owners unknowingly on wrong side of the law
# e.g., n_skip = 2 -->  train on 50%
#       n_skip = 10 --> train on 10%

n_skip = 2 # 10
n_est = 33 # number of estimators for random_forest
import sklearn
import datetime
import numpy as np
from misc import *
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

infile, out_d = "stack.bin", "output" + os.path.sep
if not exist(infile):
    a = os.system("tar xvf stack.tar.gz") # open data if not already open
if not exist(out_d): os.mkdir(out_d)

# read multispectral image
ncol, nrow, nband, data = read_binary(infile)
print("data.shape", data.shape)
bnames = band_names("stack.hdr", nband)
print("band names", bnames)

img_bands = 12
npx = nrow * ncol
ref_bands = nband - img_bands
nf_img, nf_ref = npx * img_bands, npx * ref_bands
img, ref = data[0: nf_img], data[nf_img: ]
img_names, ref_names = bnames[0: img_bands], bnames[img_bands: ] # band names

print("class names", ref_names)
img2 = bsq_to_scikit(ncol, nrow, img_bands, img)

def n_th(img, n): # take every n-th data point (data in scikit-learn expected format)
    npx, nband = img.shape
    result = np.zeros((int(math.floor(npx / n)), nband))
    for i in range(0, npx, n):
        ip = int(math.floor(i/n))
        for k in range(0, nband):
            result[ip, k] = img[i, k]
    return result

img3 = n_th(img2, n_skip)

for i in range(0, ref_bands):
    print(i)
    ref_b = bsq_to_scikit2(npx, 1, ref[(i * nrow * ncol): ((i+1) * nrow * ncol)])
    ref_b2 = n_th(ref_b, n_skip)

    rf = RandomForestClassifier(n_estimators=n_est, oob_score=True) # could crash on warn and increase # of estimators
    X, y = img3, ref_b2.ravel()
    rf.fit(X, y)

    predict = rf.predict(X)
    df = np.sum(predict - y)
    npx_t = math.floor(npx / n_skip)
    acc = 100. * ((npx_t - abs(df)) / npx_t)
    print("train%", acc)
    predict2 = rf.predict(img2)
    print("set(predict2)", set(predict2))
    df =  np.sum(predict2 - ref_b.ravel())
    acc = 100. * ((npx - abs(df)) / npx)
    print("all  %", acc)


    f, ax = plt.subplots(1, 2, sharex=True)
    ax[0].imshow(ref_b.reshape(nrow, ncol), cmap = 'binary_r')
    ax[1].imshow(predict2.reshape(nrow, ncol), cmap = 'binary_r')
    ax[0].set_title(ref_names[i])
    ax[0].set_xlabel("groundref")
    ax[1].set_xlabel("predict")
    plt.tight_layout()
    plt.show()
