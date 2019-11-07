# bcws-psu-research
software produced in collaboration with BCWS Predictive Services Unit (PSU), CITZ Data Science and Analytics Branch (DSAB) and researchers at Thompson Rivers University (TRU) for image analysis incl:

* multispectral image viewer
* Kmeans clustering applied to multispectral imagery
* invoking the following code, in python:
   * Daniel Müllner, fastcluster: Fast Hierarchical, Agglomerative Clustering Routines for R and Python, Journal of Statistical Software 53 (2013), no. 9, 1–18, URL http://www.jstatsoft.org/v53/i09/
* manipulation or visualization of class maps (i.e., truth data or classification results)
* clustering algorithm, a new minimalist implementation of
    * [Unsupervised Nonparametric Classification of Polarimetric SAR Data Using The K-nearest Neighbor Graph](http://ashlinrichardson.com/uvic/papers/2010_richardson_igarss.pdf), A. Richardson et al, proc. IEEE IGARSS, Honolulu, Hawaii, July 2010

To appear: more utilities for integrating open remotely-sensed imagery

## Direct collaborators:
* Musfiq Rahman
* Jabed Tomal
* Brady Holliday
* Dana Hicks
* Joanna Wand
### TRU Students co-supervised:
* Francesca Rammuno
* Gagan Bajwa
* Brad Crump

## requirements:
* python 2 /3 + matplotlib + numpy (for image viewer)
* gnu/g++ (for discretization algorithm)

Tested on ubuntu

## how to:
0) set up the programs:
    python setup.py
    
1) view the sample input data:

    python read_multispectral.py sentinel2_cut.bin

2) run hierarchical agglomerative clustering on sample data:

    python fast_cluster.py mS2.bin
    
3) run kmeans clustering on the sample data:

    python kmeans.py mS2.bin

## Preliminary Results

### K-means unsupervised classification
<img src="output/kmeans.bin_wheel.bin.png" width="640">

### Hierarchical Agglomerative Clustering (HAC) unsupervised classification
10 clusters: <br>
<img src="output/fastclust.bin_wheel.bin.png" width="640">

### unsupervised classification: an original method
10 clusters: <br>
<img src="output/317.lab_wheel.bin.png " width="640">

#### How the number of clusters changes by varying K (the number of K-nearest Neighbours) for the latter method
y = log(n_segments), x = number of k-nearest neighbours 
![alt text](output/nclass.csv_plot.png)

Hypothetically for a one-level analysis (non-hierarchical) taking K=100 is highly information-preserving choice, as the curve seems to depart strongly from monotonicity after K=200..

..hence K=200 or so provides efficiency without excessive info. loss
#### output formats
The clustering algorithm output is provided in two formats:

    1) Cluster labels in IEEE 32-bit Floating-point format: 0. unlabelled, labels start at 1.
        
    2) Image where the pixels are colored according to the cluster "centres" to which they're assigned

## License

Copyright 2019 Province of British Columbia

Licensed under the Apache License, Version 2.0 (the "License");
you mayn't use these files except in compliance with the License.

You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless req'd by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

See the License for specific language governing permissions
and limitations under the License.
