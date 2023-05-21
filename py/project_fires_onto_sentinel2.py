'''20230520 project fire locations onto Sentinel-2 tiles
'''
from misc import run, err, args, exists, sep
from osgeo import ogr
import sys
import os

expected_path = '/home/' + os.popen('whoami').read().strip() + '/GitHub/wps-research/py'
if os.getcwd() != expected_path:
    err('need to run this from: ' + expected_path)

rd = 'reproject'
if not exists(rd):
    os.mkdir(rd)

# shapefile_reproject.py [input shapefile] [shapefile or raster to get CRS from] [output shapefile]
'''
tiles_4326 = 'sentinel2_bc_tiles_shp/Sentinel_BC_Tiles_EPSG_4326.shp'
for f in ['prot_current_fire_points.shp', 'prot_current_fire_polys.shp']:
    base = f.split('.')[0]
    reproject = rd + sep + base + '_reproject.shp'
    if exists(reproject):
        os.remove(reproject)

    cmd = ' '.join(['shapefile_reproject.py',
                                    f,
                                    tiles_4326,
                                    reproject])
    run(cmd)
'''

def print_feature(feature):
	num_fields = feature.GetFieldCount()
	
	# Loop through all attribute fields
	for i in range(num_fields):
		field_name = feature.GetFieldDefnRef(i).GetName()
		field_value = feature.GetField(i)
		print(f"{field_name}: {field_value}")

# intersect one shapefile with another
my_tiles = {}
def shapefile_intersect(s1_path, s2_path):
    # Open the first shapefile
	s1_driver = ogr.GetDriverByName('ESRI Shapefile')
	s1_dataSource = s1_driver.Open(s1_path, 0)
	if s1_dataSource is None:
		err('Error opening:' + s1_path)

	s1_layer = s1_dataSource.GetLayer()
	s1_crs = s1_layer.GetSpatialRef()	
	# Open the second shapefile
	s2_driver = ogr.GetDriverByName('ESRI Shapefile')
	s2_dataSource = s2_driver.Open(s2_path, 0)
	if s2_dataSource is None:
		err('Error opening:' + s2_path)
	
	s2_layer = s2_dataSource.GetLayer()
	s2_crs = s2_layer.GetSpatialRef()

	if s1_crs.ExportToWkt() == s2_crs.ExportToWkt():
		print("The CRS of the two shapefiles is the same.")
	else:
		print("The CRS of the two shapefiles is different.")
		print(s1_path, s1_crs.ExportToWkt())
		print(s2_path, s2_crs.ExportToWkt())

    # Loop through features in the first shapefile
	for feature1 in s1_layer:
		# print_feature(feature1) # print(feature1)
		geometry1 = feature1.GetGeometryRef()
		# Loop through features in the second shapefile	
		for feature2 in s2_layer:
			geometry2 = feature2.GetGeometryRef()
			# Check if the two geometries intersect
			# print_feature(feature2)
			if geometry1.Intersects(geometry2):
				print("Features intersect!")
				#print_feature(feature1)
				my_fire = feature1.GetField("FIRE_NUM")
				my_tile = feature2.GetField("Name")
				if my_fire not in my_tiles:
					my_tiles[my_fire] = set()
				my_tiles[my_fire].add(my_tile)
				print("-------------------")
				#print_feature(feature2)
				#print(geometry1)
				#print(geometry2)
	# Clean up and close the shapefile data sources
	s1_dataSource = None
	s2_dataSource = None

s1 = 'reproject/prot_current_fire_polys_reproject.shp'
s2 = 'sentinel2_bc_tiles_shp/Sentinel_BC_Tiles_reprojected.shp' #tiles_4326 # 'sentinel2_bc_tiles_shp/Sentinel_BC_Tiles_EPSG_4326.shp'

shapefile_intersect(s1, s2)

for fire in my_tiles:
	print(fire, my_tiles[fire])
