'''sort a csv by selected field'''
from misc import *

if len(args) < 3:
    err('python3 csv_sort.py [input csv file name] [name of field to sort on (increasing)]')

ofn, fi = args[1] + '_sort.csv', fi
fields, data = read_csv(args[1])
lookup = {fields[i]: i for i in range(len(fields))}

try: fi = lookup[args[2]]
except: err('field not found:', args[2])

N = len(data[0])
sortd = [ [data[fi][i], i] for i in range(N)] # intentional naming
sortd.sort(reverse=True)

print("+w", ofn)
M = range(len(fields))
lines = [','.join(fields)]
for i in range(N):
    lines += [','.join([data[j][sortd[i][1]] for j in M])]
open(ofn, 'wb').write(('\n'.join(lines)).encode())

