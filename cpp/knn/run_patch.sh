rm patch.exe # catch fail
g++ patch.cpp ../misc.cpp -o patch.exe
./patch.exe stack.bin 12 3
ls -latrh
