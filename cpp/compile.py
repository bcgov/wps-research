import os
import sys

files = os.popen("ls -1 *.cpp").readlines()
files = [f.strip() for f in files]

of = open("compile.sh", "wb")
    #!/usr/bin/env bash
of.write("#!/usr/bin/env bash".encode())
for f in files:
    fn = f[:-4]
    if fn != "misc":
        s = '\ntest ! -f ' + fn + '.exe && g++ -w -O3 ' + fn + '.cpp  misc.cpp -o ' + fn + '.exe -lpthread'
        of.write(s.encode())
of.close()
