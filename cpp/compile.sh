#!/usr/bin/env bash
test ! -f band_splice.exe && g++ -w -O4 band_splice.cpp  misc.cpp -o band_splice.exe -lpthread &
test ! -f binary_confusion.exe && g++ -w -O4 binary_confusion.cpp  misc.cpp -o binary_confusion.exe -lpthread &
test ! -f binary_dilate.exe && g++ -w -O4 binary_dilate.cpp  misc.cpp -o binary_dilate.exe -lpthread &
test ! -f binary_invert.exe && g++ -w -O4 binary_invert.cpp  misc.cpp -o binary_invert.exe -lpthread &
test ! -f cat_append.exe && g++ -w -O4 cat_append.cpp  misc.cpp -o cat_append.exe -lpthread &
test ! -f class_match_onto.exe && g++ -w -O4 class_match_onto.cpp  misc.cpp -o class_match_onto.exe -lpthread &
test ! -f class_mean.exe && g++ -w -O4 class_mean.cpp  misc.cpp -o class_mean.exe -lpthread &
test ! -f class_onehot.exe && g++ -w -O4 class_onehot.cpp  misc.cpp -o class_onehot.exe -lpthread
test ! -f class_recode.exe && g++ -w -O4 class_recode.cpp  misc.cpp -o class_recode.exe -lpthread &
test ! -f class_wheel.exe && g++ -w -O4 class_wheel.cpp  misc.cpp -o class_wheel.exe -lpthread &
test ! -f cluster.exe && g++ -w -O4 cluster.cpp  misc.cpp -o cluster.exe -lpthread &
test ! -f convert_iq_to_s2.exe && g++ -w -O4 convert_iq_to_s2.cpp  misc.cpp -o convert_iq_to_s2.exe -lpthread &
test ! -f cv.exe && g++ -w -O4 cv.cpp  misc.cpp -o cv.exe -lpthread &
test ! -f hclust.exe && g++ -w -O4 hclust.cpp  misc.cpp -o hclust.exe -lpthread &
test ! -f kmeans.exe && g++ -w -O4 kmeans.cpp  misc.cpp -o kmeans.exe -lpthread &
test ! -f kmeans_multi.exe && g++ -w -O4 kmeans_multi.cpp  misc.cpp -o kmeans_multi.exe -lpthread
test ! -f multilook.exe && g++ -w -O4 multilook.cpp  misc.cpp -o multilook.exe -lpthread &
test ! -f multiply.exe && g++ -w -O4 multiply.cpp  misc.cpp -o multiply.exe -lpthread &
test ! -f nodata.exe && g++ -w -O4 nodata.cpp  misc.cpp -o nodata.exe -lpthread &
test ! -f snip.exe && g++ -w -O4 snip.cpp  misc.cpp -o snip.exe -lpthread &
test ! -f unstack.exe && g++ -w -O4 unstack.cpp  misc.cpp -o unstack.exe -lpthread &
test ! -f vri_rasterize.exe && g++ -w -O4 vri_rasterize.cpp  misc.cpp -o vri_rasterize.exe -lpthread &
wait