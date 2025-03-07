'''20211128 plot spectra, labelled by class
input: csv
  * spectra fields are identified by ending in nm
input: field label of class..
...spectra with the same label are given the same color and same key in legend'''
import os
import sys
import csv
import matplotlib
import matplotlib.pyplot as plt
from misc import read_csv
from misc import markers
from misc import colors
from misc import exist
from misc import err
args = sys.argv

if len(args) < 3:
    print("python3 csv_spectra_plot_class.py [csv input file with spectra (nm)] [selected field for legending]")

'''read the csv and locate the spectra'''
fields, data = read_csv(args[1])
nf = len(fields)  # number of fields
f_i = {fields[i]:i for i in range(nf)}

if len(args) < 3:  # call the program on all fields!
    for f in fields:
        if (f[-2:] != 'nm') and \
                (f not in ['ObjectID', 'GlobalID', 'x', 'y',
                           'ctr_lat', 'ctr_lon', 'image']):
            cmd = 'python3 ' + __file__ + ' ' + args[1] + ' ' + f
            print(cmd)
            a = os.system(cmd)
    sys.exit(1)

if args[2] not in fields:
    print("Error: field not found:", args[2])
    print(fields)
fi = f_i[args[2]]  # col index of selected field for legending
field_label = args[2].strip().replace(' ', '-')

spec_fi = []
for i in range(nf):
    if fields[i][-2:] == 'nm':
        spec_fi += [i]
print('spectra col-ix', spec_fi)
print('number of cols', len(spec_fi))

'''before we plot, code the categorical possibilities from 0 to however many there are'''
N = len(data[0]) # number of data points
print("number of data points", N)
lookup, next_i = {}, 0
for i in range(N):
    value = data[fi][i]
    if value not in lookup:
        lookup[value] = next_i
        next_i += 1
values = lookup.keys()
reverse_lookup = {lookup[x]: x for x in values}
print("lookup", lookup)   # take categorical value and encode it as int 0,1,..
print("revers", reverse_lookup)  # recover original value from int code!

def deriv(x, spectrum):
    spec_new = []
    for i in range(1, len(x)):
        spec_new.append(spectrum[i] - spectrum[i - 1])
    return x[1:], spec_new

def integ(x, spectrum):
    spec_new = [spectrum[0]]
    for i in range(1, len(x)):
        spec_new.append(spec_new[-1] + spectrum[i])
    return x, spec_new

for case in ['regular', 'derivative', 'integral']:
    '''now do the actual plotting'''
    plt.figure(figsize=(8*2.5,6*2.5))
    plt.title((case if case != 'regular' else '') + "Spectra aggregated by categorical field: " + args[2])
    plt.ylabel("Digital number")
    plt.xlabel("Date, resolution(m) and Frequency (nm)")
    # plt.gca().axes.get_yaxis().set_visible(False)

    max_y, min_y = 0, 0
    ci = 0
    for i in range(N):
        value = data[fi][i]
        spec = [float(data[j][i]) for j in spec_fi]
        for j in range(len(spec)):
            y = spec[j]
            max_y = y if y > max_y else max_y
            min_y = y if y < min_y else min_y
            ci += 1
    print("ymin", min_y, "ymax", max_y)

    used_value=set()
    for i in range(N):
        x = range(len(spec_fi))
        value = data[fi][i] # categorical value
        spectrum = [float(data[j][i]) for j in spec_fi]
        print(value, spectrum)
        if case == 'derivative':
            x, spectrum = deriv(x, spectrum)
        if case == 'integral':
            x, spectrum = integ(x, spectrum)
        plt.plot(x,
                 spectrum, # marker=markers[lookup[value]],
                 color=colors()[lookup[value]],
                 label=(value if value not in used_value else None))
        used_value.add(value)
        # don't forget to put the spectra field labels on the bottom as ticks!
    #plt.legend() # loc='lower left') # upper right')
    plt.xticks(x, [fields[i] for i in spec_fi[0:len(x)]], rotation='vertical')
    plt.legend()
    plt.tight_layout()
    fn = args[1] + "_spectra_plot_" + (case if case != 'regular' else '') + '_' + field_label + ".png"
    print("+w", fn)
    plt.savefig(fn)
