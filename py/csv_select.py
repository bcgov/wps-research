'''Revised 20230327: csv_select.py Select records from CSV based on a 1-column csv file of keys to match (on the field indicated in the header)  
e.g: 
python csv_select.py vri/VRI_KLoops.csv vri/vri_objid.csv'''
import os
import sys
import csv
from misc import args, err

if len(args) < 3:
    print('csv_select.py [csv file] [select file]')
    print('# warning: csv-header fields expected to not contain comma') 
    err('# select file is 1-col csv with keys to select from col identified by header in 1-col csv')

csv_file, select_file = args[1], args[2]
csv_f = open(csv_file)
csv_hdr = csv_f.readline().strip().split(',')
csv_i_hdr = {csv_hdr[i]: i for i in range(0, len(csv_hdr))}

select = open(select_file)
select_hdr = select.readline().strip().split(',')

if len(select_hdr) != 1:
    err("select file header length")
select_f = select_hdr[0]

if select_f not in csv_hdr:
    err("select_f not in csv_hdr")
csv_select_i = csv_i_hdr[select_f]
print("csv_select_i", csv_select_i)
select_list = []
while True:
    line = select.readline()
    if not line:
        break
    else:
        select_list.append(line.strip())

print("select_list", select_list)
of, ci = open("csv_select.csv", "wb"), 0

# write csv header
of.write((','.join(csv_hdr)).encode())

while True:
    if ci % 10000 == 0: print("ci", ci)
    line = csv_f.readline()
    if not line:
        break
    w = line.strip().split(",")

    if len(w) != len(csv_hdr):
        csv.register_dialect('my',
                     delimiter=",",
                     quoting=csv.QUOTE_ALL,
                     skipinitialspace=True)
        r = []
        reader = csv.reader([line])
        for row in reader: 
            r.append(row)
        w = r[0]
    dd = w[csv_select_i]
    if float(dd) == float(int(dd)):
        dd = str(int(dd))
    if dd in select_list:
        of.write(('\n' + line.strip()).encode())
    ci += 1
of.close()
