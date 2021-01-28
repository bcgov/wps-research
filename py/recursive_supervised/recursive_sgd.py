'''SGD classifier-- linear SVM. Try RADIAL BASIS FUNCTION SVM??? https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
cd bcws_psu_research/recursive_classifier/
mkdir out
python3 recursive_sgd.py stack.bin out/

todo: write inputs, accuracy etc, to log file!!!!!'''

import sys; sys.path.append("../py/")

import sklearn
import datetime
import numpy as np
from misc import *
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

skip_i = 7  # skip every i-th sample (10 -> 90% train, 2 -> 50% train, 4 - > 75% train)
sgd = SGDClassifier(random_state=42, verbose=False, max_iter=1000, tol=1.e-3)

if len(args) < 3:
    err("sgd [input image stacked with binary ground-ref dat] [output dir]")

img_name, out_d = args[1], args[2]
if not exist(out_d) or not os.path.isdir(out_d):
    err("output directory not found")

# add slash if not there
out_d = (out_d + os.path.sep) if out_d[-1] != os.path.sep else out_d

if not exist(args[1]):
    err('could not find input image: ' + args[1])

# read multispectral image
ncol, nrow, nband, img = read_binary(args[1])

band_names = band_names(hdr_fn(args[1])) # assume first n bands are image, next m are groundref
n = 0
print("band names:", band_names)
for i in range(0, nband):
    try:
        x = int(band_names[i].split()[1])
        n = i + 1
    except:
        break

print("number of image bands: " + str(n)); m = nband - n
print("number of groundref bands: " + str(m))

npx = nrow * ncol # number of pixels
# convert binary data to numpy format expected by sgd
img_np = bsq_to_scikit(ncol, nrow, n, img[0: n * npx])


for k in range(n, nband):
    print("k", k, "__________________________")
    ref = img[k * npx: (k + 1) * npx]
    cls_name = band_names[k]
    print("class name: " + cls_name)
    
    # if cls_name != "WATER":
    #     continue

    # count positives, negatives
    ref_count = hist(ref)
    print("ref_layer_count", ref_count)
    NP, NN = ref_count[1.], ref_count[0.]

    # force groundref map to bool (assume any merging etc. already done)
    ref = np.array(ref)
    ref = (ref != 0.)

    # convert binary data to numpy format expected by sgd
    img_np = bsq_to_scikit(ncol, nrow, nband, img)

    # sample the data by skipping every skip_i'th element
    n_samp = int(npx) - int(math.ceil(npx / skip_i))
    img_samp, ref_samp = np.zeros((n_samp, nband)), np.zeros((n_samp))
    nxt_i = 0 # index for next element to copy
    for i in range(0, npx):
        if i % skip_i != 0:
            img_samp[nxt_i, :] = img_np[i, :]
            ref_samp[nxt_i] = ref[i]
            nxt_i += 1

    # sanity check: make sure we didn't fudge our integer math
    if nxt_i != n_samp:
        print("nxt_i", nxt_i, "n_samp", n_samp)
        err("unexpected n")

    # init classifier
    sgd = SGDClassifier(random_state=1,
                        loss='modified_huber', #'log', # modified_huber',
                        penalty='l2', # "elasticnet",
                        verbose=False,
                        max_iter=1000,
                        tol=.01) # 1.e-3
    sgd.fit(img_samp, ref_samp)  # fit on sampled data
    pred = sgd.predict(img_np)  # predict on full data
    prob = sgd.predict_proba(img_np)

    # need to validate this, and count accuracy
    TP = TN = FN = FP = 0
    for i in range(npx):
        if ref[i]:
            if pred[i]:
                TP += 1
            else:
                FN += 1
        else:
            if not pred[i]:
                TN += 1
            else:
                FP += 1

    if True:
        db = [0, 1, 2] # [3, 2, 1] # default band index
        a = np.zeros((nrow, ncol, 3))
        a[:, :, 0] = img_np[:, db[0]].reshape(nrow, ncol)
        a[:, :, 1] = img_np[:, db[1]].reshape(nrow, ncol)
        a[:, :, 2] = img_np[:, db[2]].reshape(nrow, ncol)
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
        f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6) # , sharey=True)
        plt.gcf().set_size_inches(6 * 6, 6.2)

        # 1. raw image
        ax1.imshow(a)

        # 2. groundref
        ax2.imshow(ref.reshape(nrow, ncol), cmap = 'binary_r') # dont' know why colour !
        
        # 3. prediction
        ax3.imshow(pred.reshape(nrow, ncol), cmap = 'binary_r')

        # 4. probability: 0:npx because there's probabilities for two classes!
        print(prob.shape)
        ax4.imshow((prob[:, 1]).reshape(nrow, ncol), label='prob', cmap='gray') # two channels: one per class

        # 5. histogram of probability:
        hst, bins = np.histogram((prob[:, 1]).reshape(nrow, ncol), bins = 25) # 'auto')
        hst = hst.ravel().tolist()
        bins = bins.ravel().tolist()
        bins = bins[0:len(hst)] 
        print("hst", hst, "|hst|", len(hst)) # hst.shape)
        print("bins", bins, "|bins|", len(bins)) #  bins.shape)
        for i in range(0, len(hst)):
            print(i, bins[i], hst[i])
        # ax5.plot(bins[1:], hist) 
        ax5.bar(bins, hst, width = 1/25., align='edge')


        # 6. projection of fit onto seed derived from thresholding probability
    
        thres = 0.5
        '''
        ti = len(bins) - 1 # start at right of histogram

        while hst[ti - 1] > hst[ti]: # go left until reach max
            ti -= 1
        thres = bins[ti]

        while hst[ti - 1] < hst[ti]: # continue left until reach min
            ti -= 1
        thres = bins[ti]
        
        print("automatically determined probability threshold", thres) '''

        ref = ((prob[:, 1]).reshape(nrow, ncol)) 
        ref = ref.ravel()
        ref = ref > thres
        ref = [1. if i==True else 0. for i in ref]
        
        #img_samp, ref_samp = np.zeros((n_samp, nband)), np.zeros((n_samp))
        ref_samp = np.zeros(n_samp)
        nxt_i = 0 # index for next element to copy
        for i in range(0, npx):
            if i % skip_i != 0:
                # img_samp[nxt_i, :] = img_np[i, :]
                ref_samp[nxt_i] = ref[i]
                nxt_i += 1

        # sanity check: make sure we didn't fudge our integer math
        if nxt_i != n_samp:
            print("nxt_i", nxt_i, "n_samp", n_samp)
            err("unexpected n")

        sgd.fit(img_samp, ref_samp)
        pred = sgd.predict(img_np)  # predict on full data
        ax6.imshow(pred.reshape(nrow, ncol), cmap = 'binary_r')

        

        # set titles
        ax1.set_title('image')
        ax2.set_title('groundref P ' + str(NP) + ' N ' + str(NN)) #: ' + groundref_name)
        ax3.set_title('predicted' + " TP " + str(TP) + " TN " + str(TN) + " FP " + str(FP) + " FN " + str(FN) ) #: ' + groundref_name)
        ax4.set_title('probability(true)')
        plt.tight_layout()
        fn = out_d + 'plot_' + img_name + '_' + cls_name + '.png'
        print("+w", fn)
        plt.savefig(fn)

        # write stats to log file

        d = datetime.date.today()
        date_str = str(d.year) + str(d.month).zfill(2) + str(d.day).zfill(2)
        lfn = out_d + date_str + '_log.txt'

        f = open(lfn, 'ab')
        P, N = TP + FN, TN + FP
        TPR, TNR = TP / P, TN / N
        accuracy = (TP + TN) / (P + N)
        balanced_accuracy = (TPR + TNR) / 2  # https://en.wikipedia.org/wiki/Precision_and_recall 
        log_data = [os.path.basename(__file__), img_name, cls_name, fn, TP, TN, FP, FN, accuracy, balanced_accuracy]
        log_data = [str(log_d) for log_d in log_data]
        for log_d in log_data:
            if len(log_d.split(',') ) > 1:
                err('delimiter should not be present within data')
        print("write:\n\t" + str(log_data))
        f.write((','.join(log_data) + '\n').encode())
        f.close()

        # next: probability, seeded prediction
