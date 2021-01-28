## how to:
0) Set up the programs:
    python setup.py
    
1) View sample input data:

    python py/read_multispectral.py data/mS2.bin
    
2) Run K-Means clustering on the sample data:

    python py/kmeans.py data/mS2.bin

3) Run Hierarchical Agglomerative Clustering (HAC) on sample data:

    python py/fast_cluster.py data/mS2.bin
 
4) Run HAC to produce tiled seeds:
    
    python py/fast_cluster_tiling.py data/mS2.bin 50 5 24.5
    
5) Run HAC with seeds produced by 4), allowing HAC to run on larger images

    bin/hclust data/mS2.bin data/mS2.bin_label.bin 10
    
    video of clustering: https://www.youtube.com/watch?v=ooOHEubDNvM
    
## Scripts
* Codes for supervised, unsupervised and hybrid classification methods, for multispectral imagery
   * K-Means ++
   * HAC from [1]
   * Paralellized semi-supervised K-Means
   * Hierarchical Agglomerative Clustering (HAC) Scikit-Learn
   * HAC: bootstrappable, direct implementation (can initialize with HAC results on tiles)

