import os
import sys

def run(c):
    a = os.system(c)
    return a

run("make")
run("htrim2 dat/stack.bin 2. 2.")
run("./dem dat/stack.bin_ht.bin")
