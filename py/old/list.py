import os
import sys
import datetime
year = datetime.date.today().year
def incr(c): return chr(ord(c) + 1)
def decr(c): return chr(ord(c) - 1)
def cenc(s): return "".join([incr(si) for si in list(s)])
def cdec(s): return "".join([decr(si) for si in list(s)])
a = cdec("t4;00tfoujofm.qspevdut.db.njssps0")
ls = cdec("bxt!t4!mt!..op.tjho.sfrvftu!")
s2 = cdec("Tfoujofm.30")

c1 = ls + a + s2 + 'S2MSI1C/' + str(year) + '/02/09/'
c2 = ls + a + s2 + 'S2MSI2A/' + str(year) + '/02/09/'

def call(c):
    t = [x.strip() for x in os.popen(c).read().strip().split('\n')]
    print('\n'.join(t))


print(c1)
call(c1)

print(c2)
call(c2)

