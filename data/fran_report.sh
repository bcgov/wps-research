# kmeans

#python py/kmeans.py data/fran/mS2.bin  7
#python py/read_multi.py  data/fran/mS2.bin_kmeans.bin 1 
class_wheel data/fran/mS2.bin_kmeans.bin
python py/read_multi.py data/fran/mS2.bin_kmeans.bin_wheel.bin 1

python py/kmeans.py data/fran/mL8.bin 7
class_match_onto data/fran/mL8.bin_kmeans.bin data/fran/mS2.bin_kmeans.bin data/fran/mL8.bin_kmeans.bin_match.bin
class_wheel data/fran/mL8.bin_kmeans.bin_match.bin
python py/read_multi.py data/fran/mL8.bin_kmeans.bin_match.bin_wheel.bin 1

python py/kmeans.py data/fran/mS2_L8.bin 7
class_match_onto data/fran/mS2_L8.bin_kmeans.bin data/fran/mS2.bin_kmeans.bin data/fran/mS2_L8.bin_kmeans.bin_match.bin
class_wheel data/fran/mS2_L8.bin_kmeans.bin_match.bin
python py/read_multi.py data/fran/mS2_L8.bin_kmeans.bin_match.bin_wheel.bin 1

# hierarchical
python py/fast_cluster.py data/fran/mS2.bin 7 
class_match_onto data/fran/mS2.bin_fastclust.bin data/fran/mS2.bin_kmeans.bin data/fran/mS2.bin_fastclust.bin_match.bin
class_wheel data/fran/mS2.bin_fastclust.bin_match.bin
python py/read_multi.py data/fran/mS2.bin_fastclust.bin_match.bin_wheel.bin 1

python py/fast_cluster.py data/fran/mL8.bin 7
class_match_onto data/fran/mL8.bin_fastclust.bin data/fran/mS2.bin_kmeans.bin data/fran/mL8.bin_fastclust.bin_match.bin
class_wheel data/fran/mL8.bin_fastclust.bin_match.bin
python py/read_multi.py data/fran/mL8.bin_fastclust.bin_match.bin_wheel.bin 1

python py/fast_cluster.py data/fran/mS2_L8.bin 7
class_match_onto data/fran/mS2_L8.bin_fastclust.bin data/fran/mS2.bin_kmeans.bin data/fran/mS2_L8.bin_fastclust.bin_match.bin
class_wheel data/fran/mS2_L8.bin_fastclust.bin_match.bin
python py/read_multi.py data/fran/mS2_L8.bin_fastclust.bin_match.bin_wheel.bin 1




