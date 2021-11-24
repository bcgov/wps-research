import os
sep = os.path.sep
s = ['5MCP19/1/20210710/rgb.bin',
     '5MCP19/1/20210722/rgb.bin',
     '5MCP19/1/20210714/rgb.bin',
     '5MCP19/2/20210710/rgb.bin',
     '5MCP19/2/20210722/rgb.bin',
     '5MCP19/2/20210714/rgb.bin',
     '5MCP18/1/20210718/rgb.bin',
     '5MCP18/2/20210718/rgb.bin',
     '5MCP13/1/20210715/rgb.bin',
     '5MCP13/1/20210719/rgb.bin',
     '5MCP13/2/20210715/rgb.bin',
     '5MCP13/2/20210719/rgb.bin',
      '5MCP7/1/20210716/rgb.bin',
      '5MCP7/1/20210720/rgb.bin',
      '5MCP7/1/20210708/rgb.bin',
      '5MCP7/2/20210716/rgb.bin',
      '5MCP7/2/20210720/rgb.bin',
      '5MCP7/2/20210708/rgb.bin']

d = {}
# identify pairs to merge
for i in s:
    w = i.split(sep)
    beam = w[0] # beam mode
    st = w[1]  # dataset number
    date = w[2] # date
    key = beam + '_' + date# should be two sets per beam_date
    if key not in d: d[key] = []
    d[key].append(i)

c = []
for k in d:
    print(k, d[k])
    c += ['gdal_merge.py -o ' + k + '.bin -of ENVI -ot Float32 ' + (' '.join(d[k]))]

import multiprocessing as mp
def run(c):
    return os.system(c)

def parfor(my_function, my_inputs, n_thread=mp.cpu_count()): # eval fxn in parallel, collect
    pool = mp.Pool(n_thread)
    result = pool.map(my_function, my_inputs)
    return(result)

parfor(run, c, 4)  
