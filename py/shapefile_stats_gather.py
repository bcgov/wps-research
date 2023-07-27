'''20230630 after running shapefile_stats.py on multiple shapefiles,
gather the results into an HTML table, one column per file
'''
import sys
import os

lines = [x.strip() for x in os.popen('ls -1 out/*.png').readlines()]

attrs = set()
filenames = set()

lookup = {}
for line in lines:
    x = line.split('/')[1]
    # print(x)

    w = x.split('.shp_')
    fn = w[0] + '.shp'
    attr = w[1][:-4]

    attrs.add(attr)
    filenames.add(fn)

    if fn not in lookup:
        lookup[fn] = {}
    lookup[fn][attr] = line

attrs = sorted(list(attrs))
# print(attrs)
filenames = sorted(list(filenames), reverse=True)


print("<table>")

print("<tr>", end="")
print("<th>" + "Attribute" + "</th>", end="")
for f in filenames:
    print("<th>" + f + "</th>", end="")
print("</tr>")

for attr in attrs:
    print("<tr>", end="")
    print("<td>" + attr + "</td>", end="")
    for fn in filenames:
        print('<td><img src="' + lookup[fn][attr] + '"></td>', end="")
        #print(f, lookup[f])
        

    print("</tr>")

print("</table>")
