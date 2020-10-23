rm tile.exe # catch fail
g++ tile.cpp ../misc.cpp -o tile.exe
./tile.exe stack.bin 12 3
ls -latrh
