#!/usr/bin/bash
rm ./cluster.exe
./compile.sh
# ./cluster.exe  data/two_class.bin 
# ./cluster.exe  data/737x249x5.bin 
# ./cluster.exe data/249x249x5.bin
./cluster.exe data/237x201x3.bin
# ./cluster.exe data/checkers.bina

cd nearest; bin2png
