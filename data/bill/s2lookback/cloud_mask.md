Cloud Masking for Sentinel-2
---
This library is designed specifically to mask cloud from Sentinel-2 imagery.


Methodology
---
This is still in progress of research, but in general, this is how we design the algorithm.

    We use Sentinel-2 cloud probability as the source for our model.

The model is trained incrementally, as we have more and more data.

    Each time we run the model training, we will get back specs of that model, how good did it detect cloud.

    The model will be saved into a dir in the same library, can be loaded and use directly.

    The more dates the model is trained on, the more cases it can consider.


Datasets
---
We assume Sen2Cor's True Positive rate is much higher than True Negative rate.

i.e: if it marks a pixel as cloud, it's very likely that pixel is actually cloud. 

From that, it is generally safer to sample from pixels that were masked as cloud than the opposite.

    Sampling: To construct a datasets, we will sample directly from the data with a ratio of our choice.

    We should expect: a portion of each will not represent the true label.


Parameters
---





