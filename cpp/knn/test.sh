rm patch.exe & # catch fail
rm dedup.exe & 
wait
g++ patch.cpp ../misc.cpp -o patch.exe & 
g++ dedup.cpp ../misc.cpp -o dedup.exe &
wait

./patch.exe stack.bin 12 3
./dedup.exe stack.bin

