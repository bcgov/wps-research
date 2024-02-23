'''20240222 merge sheets from multiple-sheet XLSX file, into csv. Repeat the col names only once at the beginning'''
from misc import err
import pandas as pd
import sys
args = sys.argv

if len(args) < 2:
    err('xls_merge.py [.xlsx file with same col names, on top of multiple sheets]')
xl_file = pd.ExcelFile(args[1])

dfs = [[sheet_name, xl_file.parse(sheet_name)]
          for sheet_name in xl_file.sheet_names]


f0 = None # fields of first sheet
for d in dfs:
    fields = list(d[1].columns)
    if f0 is None:
        f0 = ','.join([str(x) for x in fields])
    else:
        if f0 != ','.join([str(x) for x in fields]):
            print("Error: fields in " + d[0] + " didn't match fields in " + dfs[0][0])
            sys.exit(1)

# iterate the data
d_i = 0
print(f0)
for d in dfs:
    fields = list(d[1].columns)

    nrow = d[1].shape[0]

    for i in range(1, nrow):
    
        row = list(d[1].iloc[i].values)
        print(','.join([str(x) for x in row]))
    d_i += 1
