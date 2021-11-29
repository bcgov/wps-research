'''20211128 averaging over a window, where the windowed data are from:
    raster_extract_spectra.py'''   
import os
import sys
import csv
import matplotlib
import matplotlib.pyplot as plt
args = sys.argv

from misc import read_csv
from misc import markers
from misc import colors

'''read the csv and locate the spectra'''
fields, data = read_csv(args[1])
nf = len(fields)  # number of fields
f_i = {fields[i]:i for i in range(nf)}
if args[2] not in fields:
    print("Error: field not found:", fi)
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

'''now do the actual plotting'''
plt.figure(figsize=(8*2.5,6*2.5))
plt.title("Spectra aggregated by categorical field: " + args[2])
plt.ylabel("Digital number")
plt.xlabel("Date, resolution(m) and Frequency (nm)")
plt.gca().axes.get_yaxis().set_visible(False)

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
x = range(len(spec_fi))
for i in range(N):
    value = data[fi][i] # categorical value
    spectrum = [float(data[j][i]) for j in spec_fi]
    print(value, spectrum)
    plt.plot(x,
             spectrum, # marker=markers[lookup[value]],
             color=colors[lookup[value]],
             label=(value if value not in used_value else None))
    used_value.add(value)
    # don't forget to put the spectra field labels on the bottom as ticks!
#plt.legend() # loc='lower left') # upper right')
plt.xticks(x, [fields[i] for i in spec_fi], rotation='vertical')
plt.legend()
plt.tight_layout()
fn = "spectra_plot_" + field_label + ".png"
print("+w", fn)
plt.savefig(fn)
