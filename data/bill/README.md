<h1>Sentinel 2 Researches</h1>


About Us
---
We are a team at Predictive Services Unit (PSU) working on Data Science projects at BC Wildfire Service. Some projects are very practical, we hope to get them done very early.



High Performance Computing
---
One of our biggest strengths is a High Performance Computer that can run parallel work quickly on many tasks (512 threads).

We also have NVidia GPU infrastructure, where most of computationally expensive work is done on.


s2lookback
---
This framework is developed to generate better composite sequences.

With better cloud mask (s2lookback.masking), the composite sequence (MRAP) will not have lots of artifacts (of cloud, shadow,...) left.

**The regular flow**

    It is common to have masks generated before MRAP is done. 
    
    After confirming that you are happy with the masks, mrap accepts the same raw imagery dir and the generated mask dir, it will applying the mrap algorithm (e.g fill in the cloudy pixels with the most recent non-cloud data.)

    Now you have:

        1. Raw Imagery
        2. The masks
        3. MRAP (product from 1. and 2.)

    The final stage is cloud reduction, where the model attempts to make cloud in raw imagery "thinner", so the signal will be closer to "if no cloud was there" 

    At the end, you can choose between MRAP or cloud reduced product for your analysis.


**I. s2lookback.masking**

It can be better to have a model running individually for every tile. As we give is a directory of imageries at that specific tile, it can learn to map on the characteristics of the land (which varies between tiles).

The model will be stored, it can be used to predict for other tiles (it just needs the right input features), but not recommended.

    Syntax:
    ------

    python3 -m s2lookback.masking [IMG_DIR: where all ENVI of imageries are stored] [MASK_DIR: ENVI dir of either probability or binary map of cloud, shadow, ...] --sample=500_000  --start=2025-07-30 --end=2025-09-03 --test_only_last=False


**II. s2lookback.mrap**


    Syntax:
    ------

    python3 -m s2lookback.mrap [IMG_DIR: where all ENVI of imageries are stored] [MASK_DIR: ENVI dir of either probability or binary map of cloud, shadow, ...]


**III. s2lookback.reduce_cloud (under development)**

    Syntax:
    ------

    python3 -m s2lookback.reduce_cloud [IMG_DIR: where all ENVI of imageries are stored] [MASK_DIR: ENVI dir of either probability or binary map of cloud, shadow, ...] [MRAP_DIR: where all ENVI of composites are stored]




Current Work
---

**Burn mapping**

<p align="center"> <img src="images/burn_mapping.png" width="900"> </p>

**Cloud masking/reduction and imagery enhancement**

<p align="center"> <img src="images/cloud_reduction.png" width="900"> </p>
