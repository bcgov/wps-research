# convert dbf of spatial subset of VRI (shapefile) to csv
import sys
import dbfread

ci = 0
count = {}
keys = None
keytype = {}
f = open("VRI_KLoops.csv", "wb")

for record in dbfread.DBF("VRI_KLoops.dbf"):
    if keys is None:
        keys = record.keys()
        keys = [k.replace(',', ';') for k in keys]
        f.write(",".join(keys))
    else:
        if str(keys) != str(record.keys()):
            print "Error: record length mismatch"
            sys.exit(1)

    r = []
    for k in keys:
        r.append(str(record[k]))
        if k not in count:
            count[k] = {}
            keytype[k] = type(record[k])
        value = record[k]
        if value not in count[k]:
            count[k][value] = 0
        count[k][value] += 1

    r = [rr.replace(',', ';') for rr in r]
    f.write("\n" + ','.join(r))
print "total number of records,", ci
f.close()

ci = 0
for k in keys:
    if len(count[k]) > 1:
        print len(count[k]), k, keytype[k]
        ci += 1

print "number of differentiable fields:", ci
