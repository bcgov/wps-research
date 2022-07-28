'''Should redo this script using argparse!!!!!!

Query sentinel-2 products over a given point (alternately, a place name). Later: tile-ID of interest. Restrict for time?

20220728 Need to search using poly e.g. footprint:"Intersects(POLYGON((-4.53 29.85, 26.75 29.85, 26.75 46.80,-4.53 46.80,-4.53 29.85)))" 
'''
import os
import sys
import math
foot_print = 'Intersects(51.48252764574755,-123.95386296901019)' # bc 	2022-C50155 	52.003216999999999 	-123.13905 	2022-05-05 17:23 	2022-05-30 15:01 	UC 	H 

# foot_print = 'Intersects(51.0602686,-120.9083258)' # default location: Kamloops
# VICTORIA: (48.4283334, -123.3647222)

fpfn = None
if len(sys.argv) > 1:
    if not os.path.exists(sys.argv[1]):
        import geopy # python geocoder
        from geopy.geocoders import DataBC
        geolocator = DataBC() # user_agent = "my-application")
        location = geolocator.geocode(sys.argv[1])
        print(location.address)
        print((location.latitude, location.longitude))
        foot_print = 'Intersects(' + str(location.latitude) + ',' + str(location.longitude) + ')'
    else:
        fpfn = sys.argv[1]
        foot_print = open(fpfn).read().strip()

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
    cmd = ('wget --no-check-certificate --user=' +
           user_ + ' --password=' +
           pass_ + ' --output-document=out.html ' +
           '"https://scihub.copernicus.eu/dhus/search?q=(platformname:Sentinel-2 ' +
           'AND cloudcoverpercentage:[0 TO 100] AND ' +
           'footprint:\\"' + foot_print + '\\")' + add + '"')
    return cmd

user_ = user_.strip()
pass_ = pass_.strip()

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
    ti = titles[i].strip() # need to compare this to list of files already downloaded: skip existing files!
    tw = ti.split(">")[1].split("<")[0].strip()
    zfn = ti[7:-8] + '.zip'

    # note: if you had a folder, not a directory (a directory in the case of google download): test -f should be test -d instead!!!!!!
    cmd ='test ! -f ' + zfn + ' && wget ' + ' --no-check-certificate --content-disposition --continue --user='
    cmd += (user_ + ' --password=' + pass_ + ' "' + w + '\\$value"') # + " #" + ti)
    
    if i > 0:
        f.write('\n'.encode())

    f.write(cmd.encode())

    f.write((' > ' + zfn + '_stdout.txt 2> ' + zfn + '_stderr.txt').encode())
    
    if i % 2 == 0:
        f.write(' &'.encode())

    f.write((" #" + ti).encode())

    if i % 2 == 1:
        f.write('\nwait'.encode())

f.close()
a = os.system('chmod 755 ./.sentinel2_download.sh') 
print("+w .sentinel2_download.sh")

a = os.system('cp -v .sentinel2_download.sh ' + fpfn + '_download.sh')
