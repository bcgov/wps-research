# Welcome to bcws-psu-research
Methods and Systems for Image Analysis developed in partnership with:
* Digital Platforms and Data Division (DPDD), Office of The BC Chief Information Officer (OCIO) 
* BC Wildfire Service (BCWS) Predictive Services Unit (PSU)
* Thompson Rivers University (TRU)

in support of BC Wildfire Service (BCWS) Predictive Services Unit (PSU) program. See also: [Wildfire Predictive Services](https://github.com/bcgov/wps)

## Sample results
###  Interactive mapping w BCWS FTL MVP software: end-to-end decision support system (DSS)
<img src="output/20210127_mvp.png" width="650" height="264">

### Forest change detection w Canadian Space Agency (CSA) Radarsat Constellation Mission (RCM) compact-pol SAR data
<img src="output/difference.gif" width="640">

## Features
### BCWS FTL MVP
* Exascale viewer w interactive [2] classification
   * Semi-supervised hybrid classification method
   * Decision-support system keeping humans in the loop
   * Interactively view multitemporal multispectral image stacks of size up to system limit
* Adjustable high-contrast visualization and manipulation of multispectral imagery and classification maps

## Setup:
1. Install Oracle virtualbox https://www.virtualbox.org/wiki/Downloads under VirtualBox x.x.xx platform packages
2. Create a VM inside VirtualBox by downloading https://releases.ubuntu.com/20.04/ubuntu-20.04.1-desktop-amd64.iso and using it as your "startup disk" while using VirtualBox to create a new Virtual Machine
* call your VM something creative like VM; use default options and write down your username / password 
* use the default memory settings; try to adjust them later if they're not working
* use VDI format virtual hard disk / dynamically allocated e.g. 10GB initial size
* minimal installation for Ubuntu is fine!

3. Boot your VM and install "guest additions" for your host operating system e.g. Windows host, ubuntu guest: https://www.tecmint.com/install-virtualbox-guest-additions-in-ubuntu/
4. Under "activities" in top bar, search for Terminal (run it and add to favourites)
5. Within Terminal, Navigate to your home folder:
```bash
cd ~
```
5. Create a folder called GitHub:
```bash
mkdir GitHub
```
6. Enter the folder and clone the repo:
```bash
cd GitHub
```

7. Clone the repo:
```bash
git clone git@github.com:bcgov/bcws-psu-research.git
```

8. Enter the repo:
```bash
cd bcws-psu-research
```

9. Navigate to the MVP software interface:
```bash
cd imv
```
10. Install some "standard" packages:
```bash
sudo apt install python3 python3-matplotlib python3-sklearn python3-scipy gcc g++ freeglut3-dev
```

10. Before we run it, a little housekeeping: upgrade packages
```bash
sudo apt update && sudo apt upgrade
```




* Session: always click somewhere on full-scene/ overview window, first to buffer data under it

## Collaborators and Contributors:
* Dana Hicks, BCWS
* Joanna Wand, BCWS
* Brad Martin, BCWS
* Dr. David Hill, TRU
* Dr. Musfiq Rahman, TRU
* Ashlin Richardson, OCIO


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
[2] based on gltZpr by Nigel Stewart
