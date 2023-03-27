from misc import *

# setup for python stuff
if not exists('../bin'):
    run('mkdir -p ../bin')

run('sudo apt install python3-pip python3-setuptools')
run('sudo apt install python3-gdal libgdal-dev gdal-bin python3-rasterio rasterio') #  simplekml')
run('sudo apt update && sudo apt upgrade')
run('python3 -m pip install numpy scikit-learn matplotlib alphashape descartes utm pyproj geopy')

if not exists('../deb'):
    run('mkdir -p ../deb')

cpp = os.popen("find ../cpp/*.cpp").readlines()
for c in cpp:
    c = c.strip()
    w = c.split("/")
    leaf = w[-1]
    stem = leaf.split(".")[0]
    bf = '../bin/' + stem # + '.exe'
    cmd = "g++ -w -O4 -march=native -o " + bf + " " + c + " ../cpp/misc.cpp -lpthread"  # -g 
    if stem != "misc" and not exists(bf):
        run(cmd)

py = os.popen("find ./*.py").readlines()
for p in py:
    w = p.strip().split("/")[-1]
    fn = w.split(".")[0]
    p, wf = p.strip(), 'wrap-py_' + fn + '.cpp'
    lines = ['#include<stdlib.h>',
             '#include<iostream>',
             '#include<string>',
             'using namespace std;',
             'int main(int argc, char ** argv){',
             '  string cmd("python ");',
             '  cmd += string("' + p + '");',
             '  for(int i=1; i<argc; i++){',
             '    cmd += string(" ") + string(argv[i]);',
             '  }',
             'std::cout << cmd << endl;',
             'system(cmd.c_str());',
             'return(0);',
             '}']
    bf = '../bin/' + fn #+ '.exe'
    if w != "__init__.py" and not exists(bf):
        open(wf, 'wb').write('\n'.join(lines).encode())
        cmd = 'g++ -w -O3 -o ' + bf + ' ' + wf
        run(cmd + "; rm -rf " + wf)
