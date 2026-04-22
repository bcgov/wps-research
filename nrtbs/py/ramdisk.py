'''20230601 create or destroy a RAM-disk on Linux system at mount point:
    /ram

Usage:
    python3 ramdisk.py          # create/mount ramdisk
    python3 ramdisk.py -u       # unmount/delete ramdisk and free the ram'''
from misc import run, err, args, exists
import argparse
import math
import sys
import os


def mounted():  # determine if ramdisk is mounted
    return len(os.popen('df -h | grep /ram').readlines()) > 0


def ramdisk(unmount=False):  # mount or unmount a ramdisk
    if mounted():
        if unmount:
            run('sudo umount --force /ram')
            if mounted():
                err('ramdisk.py: failed to unmount')
            else:
                print('success: ramdisk unmounted')
                sys.exit(0)
        else:
            err('ramdisk already mounted')

    # find out total ram
    w = os.popen('cat /proc/meminfo | grep MemTotal').read().strip().split()
    if w[0] != 'MemTotal:' or w[2] != 'kB':
        err('expected keywords in cat /proc/meminfo: {MemTotal, kB}')

    # use approx half of total RAM
    k = int(math.floor(1. * int(w[1])/ 2.))

    if not exists('/ram'):
        run('sudo mkdir /ram')
    run('sudo mount -t tmpfs -o rw,size=' + str(k) + 'k tmpfs /ram')

    if mounted():
        print('Congrats: ram disk mounted at /ram')
    else:
        err('ramdisk at /ram failed to mount')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u",
                        "--unmount",
                        action="store_true",
                        help="set this flag to unmount")

    args = parser.parse_args()
    ramdisk(unmount=args.unmount)
