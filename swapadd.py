# not quite complete:
# (in progress): a program to add swap space, to increase memory that
# can be allocated (as RAM would).. Simulate larger ram.

#!/usr/bin/python
import ctypes
import sys

size = int(sys.argv[1])
class MemoryTest(ctypes.Structure):
	_fields_ = [  ('chars' , ctypes.c_char*size * 1024*1024 ) ]

try:
	test = MemoryTest()
	print('success => {0:>4}MB was allocated'.format(size) )
except:
	print('failure => {0:>4}MB can not be allocated'.format(size) )


'''
free -m shows memory use
'''

'''
# make a swap file of size 750MB
sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=750
sudo chmod 600 /var/swap.1

main@work:~# sudo /sbin/mkswap /var/swap.1
 Setting up swapspace version 1, size = 750 MiB (786427904 bytes)
 no label, UUID=a6de..
 main@work:~# sudo /sbin/swapon /var/swap.1

after doing this, now try to allocate more memory:
    should be able to do it!!!!!!


swapon -s  # see swap files

'''
