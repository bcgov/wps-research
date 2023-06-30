import matplotlib.pyplot as plt
from misc import err, exists
from osgeo import ogr
import numpy as np
import sys
import os

shapefile_path = sys.argv[1] # open shapefile

if not exists('out'):
    os.mkdir('out')

if not exists(shapefile_path):
    err("could not find input file:", shapefile_path)
driver = ogr.GetDriverByName('ESRI Shapefile')
dataset = driver.Open(shapefile_path, 0)  # 0 means read-only mode

types = set()
values = {}

layer = dataset.GetLayer()
feature = layer.GetNextFeature()  # iterate features

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

# Close the shapefile
dataset = None


for k in values:
    print(values[k])

    labels = list(values[k].keys())
    n_keys = len(labels)

    has_none = False
    stuff = [[labels[i], values[k][labels[i]]] for i in range(len(labels))]
    for i in range(len(stuff)):
        if labels[i] == None:
            has_none = True

    if has_none:
        for i in range(len(stuff)):
            stuff[i][0] = str(stuff[i][0])

    # print(stuff)
    stuff = sorted(stuff)
    labels = [stuff[i][0] for i in range(len(stuff))]
    counts = [stuff[i][1] for i in range(len(stuff))]

    # n_keys = len(labels)
    # print(k, n_keys)
    # print("labels", labels)
    # print("counts", counts)

    if n_keys > 50:
        print(values[k])

    s_f = .5
    plt.subplots(figsize=(12 * s_f, 8* s_f)) # 11 * s_f, 8.5 * s_f))
    # if all(isinstance(item, str) for item in labels):  # bar chart for categorical
    #     plt.bar(labels, counts)
    #     plt.xlabel('Labels')
    #     plt.ylabel('Counts')
    #else:
        # N = 10  # histogram for numerical
        # hist, bin_edges = np.histogram(labels, bins=N, range=(counts[0], counts[-1]), weights=counts)
        # plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
        # plt.xlabel('Value')
        # plt.ylabel('Counts')
    plt.bar([str(x) for x in labels], counts)
    plt.xlabel('Labels')
    plt.ylabel('Counts')        

    plt.title(' '.join(["FILE=" + sys.argv[1], "ATTR=" + str(k)]))
    plt.xticks(rotation='vertical')

    if len(labels) > 10:
        plt.xticks(fontsize=7)
    elif len(labels) > 20:
        plt.xticks(fontsize=1)
    else:
        plt.xticks(fontsize=10)

    plt.tight_layout()
    fn = 'out/' + sys.argv[1] + "_" + str(k) + '.png'
    print("+w", fn)
    plt.savefig(fn) #, dpi = 1200)
    # plt.show()
print(types)



# example attributes:
# {'fid': 976669.0, 'OBJECTID': 5956970.0, 'FEATURE_ID': 3927296, 'MAP_ID': '094H001', 'POLYGON_ID': 149, 'OPENING_IN': 'N', 'OPENING_SO': '4', 'OPENING_NU': None, 'FEATURE_CL': 843, 'INVENTORY_': 'V', 'POLYGON_AR': 6.9, 'NON_PRODUC': None, 'NON_PROD_1': None, 'INPUT_DATE': None, 'COAST_INTE': 'I', 'SURFACE_EX': 'N', 'MODIFYING_': 'N', 'SITE_POSIT': 'F', 'ALPINE_DES': 'N', 'SOIL_NUTRI': 'B', 'ECOSYS_CLA': '0', 'BCLCS_LEVE': 'V', 'BCLCS_LE_1': 'N', 'BCLCS_LE_2': 'U', 'BCLCS_LE_3': 'SL', 'BCLCS_LE_4': 'OP', 'INTERPRETA': '2003/01/01 00:00:00.000', 'PROJECT': 'SIKANNI_VRI', 'REFERENCE_': 1999, 'SPECIAL_CR': None, 'SPECIAL__1': None, 'INVENTOR_1': 72, 'COMPARTMEN': 57, 'COMPARTM_1': None, 'FIZ_CD': 'L', 'FOR_MGMT_L': 'Y', 'ATTRIBUTIO': '1999/01/01 00:00:00.000', 'PROJECTED_': '2022/12/31 00:00:00.000', 'SHRUB_HEIG': 0.400000005960464, 'SHRUB_CROW': 30, 'SHRUB_COVE': '5', 'HERB_COVER': 'HG', 'HERB_COV_1': '5', 'HERB_COV_2': 50, 'BRYOID_COV': 15, 'NON_VEG_CO': None, 'NON_VEG__1': None, 'NON_VEG__2': None, 'NON_VEG__3': None, 'NON_VEG__4': None, 'NON_VEG__5': None, 'NON_VEG__6': None, 'NON_VEG__7': None, 'NON_VEG__8': None, 'LAND_COVER': 'SL', 'EST_COVERA': 100, 'SOIL_MOIST': '4', 'LAND_COV_1': None, 'EST_COVE_1': None, 'SOIL_MOI_1': None, 'LAND_COV_2': None, 'EST_COVE_2': None, 'SOIL_MOI_2': None, 'AVAIL_LABE': 97, 'AVAIL_LA_1': 97, 'FULL_LABEL': '149\\Sw\\830-10/0\\hg,sl,by', 'LABEL_CENT': 1251595, 'LABEL_CE_1': 1351129, 'LABEL_HEIG': 120, 'LABEL_WIDT': 144, 'LINE_1_OPE': None, 'LINE_1_O_1': None, 'LINE_2_POL': '149', 'LINE_3_TRE': 'Sw', 'LINE_4_CLA': '830-10/0', 'LINE_5_VEG': 'hg,sl,by', 'LINE_6_SIT': None, 'LINE_7_ACT': None, 'LINE_7A_ST': None, 'LINE_7B_DI': None, 'LINE_8_PLA': None, 'PRINTABLE_': 'Y', 'SMALL_LABE': '149', 'OPENING_ID': None, 'ORG_UNIT_N': 1825, 'ORG_UNIT_C': 'DPC', 'ADJUSTED_I': 'N', 'BEC_ZONE_C': 'BWBS', 'BEC_SUBZON': 'mk', 'BEC_VARIAN': None, 'BEC_PHASE': None, 'CFS_ECOZON': 9, 'EARLIEST_N': None, 'EARLIEST_1': None, 'STAND_PERC': None, 'FREE_TO_GR': 'N', 'HARVEST_DA': None, 'FEATURE_AR': 69415.7127, 'FEATURE_LE': 1240.4043000000001, 'GEOMETRY_A': 0.0, 'GEOMETRY_L': 0.0, 'Shape_Leng': 1240.4042731549239, 'Shape_Area': 69415.71268886463}
