from py.misc import *
s2 = '20190926kamloops_data_subs350/S2A.bin_4x.bin_sub.bin_subs.bin'
df = open('data.txt').readlines()
for d in df:
    d= d.strip()
    lines = os.popen("python py/class_count.py " + d.strip()).readlines() # don't forget to add full-introspect
    #if len(lines) == 9:
    #    for i in range(0, len(lines)):
    #        print i, lines[i].strip()

    n_label = lines[5].strip().split(",")[1]
    ds = d.split("/")[1].split("_")[0].split(".")[0].upper()
    print n_label, ds


# confusion matrix between labels?


# calculate confusion matrix, for binary classes matched with truth classes?
