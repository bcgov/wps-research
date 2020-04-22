# don't forget to modify this script to restrict for time, need to be able to fetch relevant records that we've not yet considered

'''
  query sentinel-2 products over a given:
    1) point-- lat, long
    2) a TILE ID of interest?
'''
import os
import sys
import math

no_clobber = None
if len(sys.argv) > 1:
    no_clobber = ' -nc '

# save username and password to files:
user_, pass_ = None, None
if not os.path.exists('./.user'):
    user_ = input('please enter your copernicus username:').strip()
    open('./.user', 'wb').write(user_.encode())
    print('username written to ./.user')
else:
    user_ = open('./.user', 'rb').read().decode()

if not os.path.exists('./.pass'):
    pass_ = input('please enter your copernicus password:').strip()
    open('./.pass', 'wb').write(pass_.encode())
    print('password written to ./.pass')
else:
    pass_ = open('./.pass', 'rb').read().decode()

def c(add= ''):
    cmd = ('wget --no-check-certificate --user=' + user_ + ' --password=' + pass_ +
           ' --output-document=out.html "https://scihub.copernicus.eu/dhus/search?q=(platformname:Sentinel-2 AND cloudcoverpercentage:[0 TO 5] AND footprint:\\"Intersects(51.0602686,-120.9083258)\\")' + add + '"')
    return cmd

# get the first page
cmd = c('&rows=100')
print(cmd)
r = os.popen(cmd).read()

# read in result
dat = open('out.html').read()
a = os.system('mv out.html out0.html')
lines, lines_all = dat.strip().split('\n'), []
lines_all = [*lines_all, *lines]

row_per_page = 100
n_results = -1
# extract results count: <opensearch:totalResults>563</opensearch:totalResults>
for i in range(0, len(lines)):
    line = lines[i].strip()
    if line[0:25] == '<opensearch:totalResults>':
        # sanity check
        if line[-26:] != '</opensearch:totalResults>':
            err('expected end tag')
        w = line.split('<opensearch:totalResults>')[1]
        w = w.split('</opensearch:totalResults>')[0]
        n_results = int(w)

dats = []
dats.append(dat)

# get the other pages
n_page = int(math.ceil(n_results / row_per_page))
for i in range(1, n_page):
    print(row_per_page * i)
    cmd = c('&rows=100&start=' + str(row_per_page * i))
    r = os.popen(cmd).read()

    dat = open('out.html').read()
    dats.append(dat)
    a = os.system('mv -v out.html out' + str(i) + '.html')
    
    lines = dat.strip().split('\n')
    lines_all = [*lines_all, *lines]   

a = os.system('chmod 777 out*.html')

print('+w out_all.html')
open('out_all.html', 'wb').write('\n'.join(lines_all).encode())

lst = set()
# index from 1 [1:] to exclude title of search page 
titles = os.popen('grep "<title>" out_all.html').readlines()[1:]
for t in titles:
    t = t.strip()
    lst.add(t)

f = open('out_titles.txt','wb')
i = 0
for t in lst:
    if i == 0:
        f.write(t.encode())
    else:
        f.write(('\n' + t).encode())
    i += 1
f.close()

# open file to record the download commands
f = open("./.sentinel2_download.sh", "wb")

# download the data
links = os.popen('grep alternative out_all.html').readlines()
for i in range(0, len(links)):
    link = links[i]
    w = link.strip().split('<link rel="alternative" href="')[1].split('"/>')[0]
    ti = titles[i]
    tw= ti.strip().split(">")[1].split("<")[0].strip()
    cmd = 'wget ' + no_clobber + ' --content-disposition --continue --user=' + user_ + ' --password=' + pass_ + ' "' + w + '\\$value"' + " #" + titles[i].strip()
    if i > 0:
        f.write('\n'.encode())
    f.write(cmd.encode())
f.close()
print("+w .sentinel2_download.sh")
