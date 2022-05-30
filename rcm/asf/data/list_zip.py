import os
import sys
lines = open("L1.txt").readlines()
user = open(".user").read().strip()
pwd = open(".pass").read().strip()

for line in lines:
    f = line.strip()
    w = f.split(">")
    for i in range(len(w)):
        if w[i][0:4] =='http':
            x = w[i].split('<')[0]
            cmd = ('wget -c --http-user="' + user + "' --http-password='" + pwd + "' " + x)
            print(cmd)
            a = os.system(cmd)
