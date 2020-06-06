**Random forest demo**
* [Source Code rf_stack.py](rf_stack.py)
* S2 image and BCGW groundref are in stack.tar.gz which rf_stack.py will auto-extract (isn't it fun to read data without a driver?)
* 20% of pixels were selected for training
* seven estimators were used
* surprisingly, the qualitative features of the ground-ref data seem to be decently recovered..

To run:
```
python3 -m "pip install scikit-learn"
python3 rf_stack.py
```
![BROADLEAF](output/BROADLEAF.png)
![CONIFER](output/CONIFER.png)
![CUTBL](output/CUTBL.png)
![EXPOSED](output/EXPOSED.png)
![HERB_GRAS](output/HERB_GRAS.png)
![MIXED](output/MIXED.png)
![RIVER](output/RIVER.png)
![SHRUB](output/SHRUB.png)
![WATER](output/WATER.png)

