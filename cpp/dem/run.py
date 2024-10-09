import os
import sys

def run(c):
    a = os.system(c)
    return a

if not os.path.exists('dem'):
    run("make")

if not os.path.exists('dat/stack.bin_ht.bin'):
    run("htrim2 dat/stack.bin 2. 2.")

run("./dem dat/stack.bin_ht.bin")
