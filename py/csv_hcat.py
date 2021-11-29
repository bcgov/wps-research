'''horizontally concatenate two csv files'''
import os
import sys
from misc import read_csv
from misc import exist
from misc import err
args = sys.argv

if len(args) < 3:
    err('python3 csv_hcat.py [csv file 1] [csv file 2]')

'''read the csv and locate the spectra'''
lines1 = [x.strip() for x in open(args[1]).readlines()]
lines2 = [x.strip() for x in open(args[2]).readlines()]
if len(lines1) != len(lines2):
    err('different number of lines per input file')

ofn = args[1] + '_hcat.csv'
lines = [lines1[i] + ',' + lines2[i] for i in range(len(lines1))]
print('+w', ofn)
open(ofn, 'wb').write(('\n'.join(lines)).encode())

