import os
import sys
args = sys.argv
from envi import envi_header_modify

if __name__ == '__main__':
    envi_header_modify(args)    
