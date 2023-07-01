'''20230630 
Given two shapefile (A) and (B) which are assumed to have the same (mostly the same) attributes and for this application, (A) is in fact a small clipped area of the larger one (B):

Produce a third shapefile (C) which has a numerical value expresssing closeness between the same polygonal footprint's attributes vs. the aggregated attributes over the shapefile (A)
'''
use_mode = True  # select to use only most occuring outcome (per attribute) w.r.t reference sample
avoid = set(["ATTRIBUTIO", "AVAIL_LABE", "AVAIL_LA_1", "COMPARTMEN",
             "EARLIEST_1", "FEATURE_A", "FEATURE_CL", "FEATURE_ID",
             "FEATURE_LE", "FIZ_CD", "FULL_LABEL", "GEOMETRY_A",
             "GEOMETRY_L", "HARVEST_DA", "INPUT_DATE", "INTERPRETA",
             "LABEL_CENT", "LABEL_CE_1", "LINE_2_POL", "LINE_6_SIT",
             "LINE_7A_ST", "LINE_8_PLA" , "MAP_ID", "NON_PRODUC",
             "NON_PROD_1", "NON_VEG_6", "NON_VEG_7", "NON_VEG_8",
             "OBJECTID", "OPENING_ID", "ORG_UNIT_C", "ORG_UNIT_N",
             "POLYGON_AR", "POLYGON_ID", "PRINTABLE_", "PROJECT",
             "PROJECTED", "REFERENCE_", "SMALL_LABE", "SPECIAL_CR",
             "SPECIAL__1", "SHAPE_AREA", "Shape_Leng", "fid"])
import matplotlib.pyplot as plt
from misc import err, exists
from osgeo import ogr
import numpy as np
import sys
import os

shapefile_path1, shapefile_path2 = sys.argv[1], sys.argv[2] # B) 
output_shapefile_fn = shapefile_path1 + "_onto_" + shapefile_path2 + ".shp"
print("+w", output_shapefile_fn)
if not exists(shapefile_path1) or not exists(shapefile_path2):
    err("please check input files")

driver = ogr.GetDriverByName('ESRI Shapefile')
dataset = driver.Open(shapefile_path1, 0)  # 0 means read-only mode

types, values = set(), {}
layer = dataset.GetLayer()
feature = layer.GetNextFeature()  # iterate features for shapefile 1.

while feature is not None:  # attributes of the feature
    attributes = feature.items()

    for k in attributes:
        types.add(type(attributes[k]))
        v = attributes[k]
        
        if k not in values:
            values[k] = {}
        
        if v not in values[k]:
            values[k][v] = 0

        values[k][v] += 1
    # geometry = feature.GetGeometryRef() # print(geometry.ExportToWkt()) 
    feature = layer.GetNextFeature()  # next feature

dataset = None # close shapefile

# normalize the counts to add to 1 for each attribute
for k in values:  # normalize counts to add to 1 for each atribute?
    total = 0.
    
    for v in values[k]:
        total += values[k][v]
    
    for v in values[k]:
        values[k][v] /= total  # make it add to 1.

if use_mode: # select most frequent outcome only
    for k in values:
        max_c, max_v = None, None
        
        for v in values[k]:
            if max_c is None:
                max_c, max_v = v, values[k][v]
            
            if values[k][v] > max_v:
                max_c, max_v = v, values[k][v]
        values[k] = {max_c: 1.}

# open the second shapefile
dataset2 = driver.Open(shapefile_path2, 0)  # second shapefile, 0 is Read-only mode
layer2 = dataset2.GetLayer()
feature2 = layer2.GetNextFeature()  # iterate features for shapefile 1.
feature_count = layer2.GetFeatureCount()
spatial_ref = layer2.GetSpatialRef()

output_shapefile = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(output_shapefile_fn)  # output shp to write
output_layer = output_shapefile.CreateLayer('metric', geom_type=ogr.wkbPolygon,  srs=spatial_ref)

new_field = ogr.FieldDefn('metric', ogr.OFTReal)  # new attribute field for output shp
output_layer.CreateField(new_field)

ci = 1
while feature2 is not None:  # attributes of the features
    values2 = {}  # vector for this feature only, not aggregate
    attributes2 = feature2.items()
    for k in attributes2:
        v = attributes2[k]
        
        if k not in values2:
            values2[k] = {}
        
        if v not in values2[k]:
            values2[k][v] = 0
        
        values2[k][v] += 1.
    # geometry = feature.GetGeometryRef() # print(geometry.ExportToWkt()) 

    metric, n_terms = 0., 0.
    
    for k in values:
        if k in avoid:  # skip comparing "avoid" attributes
            continue 

        if k in values2:
            for v in values2[k]:

                if v in values[k]:
                    metric += values[k][v] * values2[k][v]   # always <= 1.
                    n_terms += 1.
                    
                    if values2[k][v] != 1.:
                        err("this quantity should always be 1.")
    metric /= n_terms  # total value should be <= 1.
    geometry = feature2.GetGeometryRef()
    output_feature = ogr.Feature(output_layer.GetLayerDefn())
    output_feature.SetGeometry(geometry)
    output_feature.SetField('metric', metric)  # numeric field in new shapefile
    output_layer.CreateFeature(output_feature)
    ci += 1  # progress meter
    if ci % 100 == 0:
        print("%", 100. * ci / feature_count, " ",  ci, "of", feature_count)
    feature2 = layer2.GetNextFeature()  # next feature

dataset2 = None  # close shapefiles
output_shapefile = None
print("+w", output_shapefile_fn)
