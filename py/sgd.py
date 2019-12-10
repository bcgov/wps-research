'''SGD classifier-- linear SVM?
'''
import sklearn
import numpy as np
from misc import *
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

skip_i = 2 # skip every i-th sample (10 -> 90% train)
sgd = SGDClassifier(random_state=42, verbose=False, max_iter=1000, tol=1.e-3)

if len(args) < 3:
    err("sgd [input image] [binary ground-reference data]")

def varname(s):
    s = s.strip()
    w = s.split(os.path.sep)[-1]
    w = w.split('.')[0]
    w = w.replace('_', '-')
    print(w)
    return(w)

# data names
img_name = varname(args[1])
cls_name = varname(args[2])

# read multispectral image
ncol, nrow, nband, img = read_binary(args[1])

# binary class map to predict
nc2, nr2, nb2, ref = read_binary(args[2])

# assert image and groundref classes match
if ncol != nc2 or nrow != nr2:
    err("image dimensions mismatch")
npx = nrow * ncol # number of pixels

# force groundref map to bool (assume any merging etc. already done)
ref = np.array(ref)
ref = (ref != 0.)

# convert binary data to numpy format expected by sgd
img_np = bsq_to_scikit(ncol, nrow, nband, img)

# sample the data by skipping every skip_i'th element
n_samp = int(npx) - int(math.floor(npx / skip_i))
img_samp, ref_samp = np.zeros((n_samp, nband)), np.zeros((n_samp))
nxt_i = 0 # index for next element to copy
for i in range(0, npx):
    if i % skip_i != 0:
        img_samp[nxt_i, :] = img_np[i, :]
        ref_samp[nxt_i] = ref[i]
        nxt_i += 1

# sanity check: make sure we didn't fudge our integer math
if nxt_i != n_samp:
    err("unexpected n")

# init classifier
sgd = SGDClassifier(random_state=42,
                    verbose=False,
                    max_iter=1000,
                    tol=1.e-3)
sgd.fit(img_samp, ref_samp)  # fit on sampled data
pred = sgd.predict(img_np)  # predict on full data


if True:
    a = np.zeros((nrow, ncol, 3))
    a[:, :, 0] = img_np[:, 3].reshape(nrow, ncol)
    a[:, :, 1] = img_np[:, 2].reshape(nrow, ncol)
    a[:, :, 2] = img_np[:, 1].reshape(nrow, ncol)
    a = (a - np.min(a)) / np.max(a)

    for i in range(0, 3):
        d = a[:, :, i]
        npx = nrow * ncol
        values = d.reshape(np.prod(d.shape)).tolist()
        values.sort()
        mn = values[int(math.floor(float(npx) * 0.01))]
        mx = values[int(math.floor(float(npx) * 0.99))]
        rng = mx - mn
        a[:, :, i] -= mn
        if rng > 0.:
            a[:, :, i] /= rng
        (a[:, :, i])[a[:, :, i] < 0.] = 0.
        (a[:, :, i])[a[:, :, i] > 1.] = 1.


    plt.close('all')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    plt.gcf().set_size_inches(6 * 3, 6.) #, 7. * float(lines) / float(samples))
    ax1.imshow(a)
    ax2.imshow(ref.reshape(nrow, ncol), cmap = 'binary_r') # why we have to reverse colourmap, don't know!
    ax3.imshow(pred.reshape(nrow, ncol), cmap = 'binary_r') #, cmap='binary')
    ax1.set_title('image')
    ax2.set_title('reference') #: ' + groundref_name)
    ax3.set_title('predicted') #: ' + groundref_name)
    plt.tight_layout()
    fn = 'plot_' + img_name + '_' + cls_name + '.png'
    print("+w", fn)
    plt.savefig(fn)
