#!/usr/bin/bash
rm ./cluster.exe
./compile.sh
./cluster.exe  data/two_class.bin > tmp.txt # 249x249x5.bin
# ./cluster.exe data/737x249x5.bin
