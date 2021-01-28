# bcws-psu-research
Methods and Systems for Image Analysis developed in partnership with:
* Digital Platforms and Data Division (DPDD), Office of The BC Chief Information Officer (OCIO) 
* BC Wildfire Service (BCWS) Predictive Services Unit (PSU)
* Thompson Rivers University (TRU)

## Sample results
### Interactive mapping with BCWS FTL MVP software: end-to-end decision support system (DSS)
<img src="output/20210127_mvp.png" width="817" height="332">

### Forest change detection product using RCM
<img src="output/difference.gif" width="640">

## Features
### BCWS FTL MVP
* Exascale viewer with interactive classification
   * Semi-supervised hybrid classification method
   * Decision-support system keeping humans in the loop
   * Interactively view multitemporal multispectral image stacks of size up to system limit
* Adjustable high-contrast visualization and manipulation of multispectral imagery and classification maps

## Setup:
- Install Oracle virtualbox

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

## Collaborators and Contributors:
* Dana Hicks, BCWS
* Joanna Wand, BCWS
* Brad Martin, BCWS
* Dr. Musfiq Rahman, TRU
* Dr. David Hill, TRU

### TRU Computing Science ML/AI Students Co-supervised:
* Gagan Bajwa
* Brad Crump
* Francesca Rammuno

## Alumni:
* Brady Holliday, BCWS
* Jabed Tomal, TRU

## License

Copyright 2021 Province of British Columbia

Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.

You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless req'd by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied.

See the License for specific language governing permissions
and limitations under the License.

## References
[1]  Daniel Müllner, fastcluster: Fast Hierarchical, Agglomerative Clustering Routines for R and Python, Journal of Statistical Software 53 (2013), no. 9, 1–18, URL http://www.jstatsoft.org/v53/i09/
