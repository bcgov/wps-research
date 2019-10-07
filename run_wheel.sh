rm class_wheel.exe
#dent class_wheel.cpp
g++ -w -O3 -march=native -o class_wheel.exe class_wheel.cpp misc.cpp -lpthread
./class_wheel.exe 20190926kamloops_data/WATERSP.tif_project_4x.bin_sub.bin


