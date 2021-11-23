# rm cluster
# vim cluster.cpp
# dent cluster.cpp
# ./compile.sh
# ./cluster mS2.bin
# Rscript --vanilla csv_plot.R nclass.csv


# clean the programs folder
#rm -f bin/*
python setup.py

# clean the data folder
#rm -f data/*
#tar xvf data.tar.gz 

# run clustering
if [ 0 == 1 ]
then
  bin/kmeans data/mS2.bin 10
  bin/fast_cluster data/mS2.bin 10
else
  echo "not running clustering"
fi

# remove any existing output png image files (web-friendly format)
rm data/*.png

# recode data to use a more intelligible format (float rep of integer, instead of close to epsilon)
class_recode data/mS2.bin_fastclust.bin
class_recode data/mS2.bin_kmeans.bin

# tidy up file names
mv data/mS2.bin_fastclust.bin_recode.bin data/mS2.bin_fastclust.bin
mv data/mS2.bin_kmeans.bin_recode.bin data/mS2.bin_kmeans.bin

# count the classes in class map (s)
if [ 0 == 1]
then
  class_count data/mS2.bin_fastclust.bin
  class_count data/mS2.bin_kmeans.bin
else
  echo "not performing class counts"
fi

# use wheel colour encoding for the two outputs, neither of these will be co-encoded however
class_wheel data/mS2.bin_fastclust.bin
class_wheel data/mS2.bin_kmeans.bin

# make png image format version of the above

python py/read_multi.py data/mS2.bin_fastclust.bin_wheel.bin 1
python py/read_multi.py data/mS2.bin_kmeans.bin_wheel.bin 1

# match one file's class map onto the class map of another image, and convert the output to png format, via the wheel encoding operator
class_match_onto data/mS2.bin_fastclust.bin data/mS2.bin_kmeans.bin data/fastclust_onto_kmeans.bin
class_wheel data/fastclust_onto_kmeans.bin
python py/read_multi.py data/fastclust_onto_kmeans.bin_wheel.bin 1

# plot number of clusters (logged) against KNN, in KGC algorithm
cd data/cluster; Rscript ../../R/csv_plot.R nclass.csv
cd ../../;

# take the KGC output, and convert the wheel representation to png (need to verify this step)
python py/read_multi.py data/kgc324.bin_wheel.bin   1

# match KCG class map onto kmeans encoding, then convert to png
class_match_onto data/kgc324.bin data/mS2.bin_kmeans.bin data/kgc_onto_kmeans.bin
class_wheel data/kgc_onto_kmeans.bin
python py/read_multi.py data/kgc_onto_kmeans.bin_wheel.bin

exit











  rm -f data/mS2.bin_kmeans.bin_match.bin
  bin/class_match_onto data/mS2.bin_kmeans.bin    data/mS2.bin_fastclust.bin data/mS2.bin_kmeans.bin_match.bin
  
  rm -f data/mS2.bin_fastclust.bin_match.bin
  bin/class_match_onto data/mS2.bin_fastclust.bin data/mS2.bin_kmeans.bin    data/mS2.bin_fastclust.bin_match.bin

  bin/class_wheel data/mS2.bin_kmeans.bin
  #bin/class_wheel data/mS2.bin_fastclust.bin
  #bin/class_wheel data/mS2.bin_kmeans.bin_match.bin
  bin/class_wheel data/mS2.bin_fastclust.bin_match.bin

  rm data/*.png
  bin/read_multi data/mS2.bin_fastclust.bin_match.bin_wheel.bin 1
  #bin/read_multi data/mS2.bin_kmeans.bin_match.bin_wheel.bin 1 
  #bin/read_multi data/mS2.bin_fastclust.bin_wheel.bin 1 
  bin/read_multi data/mS2.bin_kmeans.bin_wheel.bin 1


