import os
import sys

def run(c):
    a = os.system(c)
    return a

run("make")
run("./dem dat/stack_scale.bin")
