'''created a sanitized csv without commas (replace with semicolon)'''
from misc import *

if len(args) < 2:
    err('python3 csv_sanitize.py [input csv file name]')

ofn = args[1] + '_sanitize.csv'
fields, data = read_csv(args[1])
fields = [x.strip().replace(',', ';') for x in fields]  # sanitise fields
lookup = {fields[i]: i for i in range(len(fields))}
 
'''sanitise the data'''
for j in range(len(data)):
    data[j] = [x.strip().replace(',', ';') for x in data[j]]

N = len(data[0])
print("+w", ofn)
M = range(len(fields))
lines = [','.join(fields)]
for i in range(N):
    lines += ','.join([data[j][i] for j in range(M)])
open(ofn, 'wb').write(('\n'.join(lines)).encode())

