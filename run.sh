rm cluster.exe
vim cluster.cpp
dent cluster.cpp
./compile.sh
./cluster.exe mS2.bin
Rscript --vanilla csv_plot.R nclass.csv
