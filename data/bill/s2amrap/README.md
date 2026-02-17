Adjusted Most Recent Available Pixel (MRAP)
---

This library helps fill the pixels with objectives like cloud/haze/... with the previous recent pixel, but lighting will be adjusted.


Modules
---

1. data ordering

+ Simply order filename in the 2 folders by datetime (XML file).
I.e: There will be S2A and S2B with the same dates next to each others

+ Dictionary to record if there is cloud mask file for that image file -> They must have the same meta acquisition time.

    Record file name e.g date['2025-09-02'] = {'L1': ..., 'cloud': ...}

    When the process is over, print the list of dates that do not have cloud mask and raise error.

+ If processes is complete, return dictionary sorted by datetime.


2. amrap 

+ Receives data, ordering and check using Data Ordering

+ Mrapping

    Hyperparameters

    + min_cloud_prop_trigger: float [0, 1]

        The minimum percentage of pixels that are cloud-masked to actually mrap on it (can set to 1% for example, so if cloud content is actually less than 1% it will not mrap, set to 0 means just 1 pixel of cloud will trigger the algorithm).

        Default: 0.01


    + max_mrap_days: int

        Only mrap within this number of days. Prepare the dates before hand. 

        Default: 7 (1 week)


    + cloud_threshold: float [0, 1]

        Each pixel has a probability of being cloud, if it's less than or equal to the threshold, then it will not be flagged as concerning and we treat it as non cloud pixel.

        Default: 0.005


    Work flow:

        1. We already have the dictionary of filenames, so extract date list

        2. For date from 2nd date list (mrapping on 1st one is impossible), we will start to refer back.

            

