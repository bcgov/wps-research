'''20241205 convert EODMS results file ( geojson ) to csv..
..since the geojson results file is very hard to read.
'''
import json 
data = json.load(open('query_results.geojson'))

stuff = {}

for f in data['features']:
    for k in f['properties'].keys():
        if k not in stuff:
            stuff[k]= [ k ]
        stuff[k] += [f['properties'][k]]

keys = list(stuff.keys())
M, N = len(keys), len(stuff[keys[0]])

data = []
for key in stuff.keys():
    data += [ stuff[key] ]

data2 = [[] for i in range(N)]
for j in range(M):
    for i in range(N):
        data2[i] += [data[j][i]]

for d in data2:
    print(','.join([str(x) for x in d]))
