'''
Automatically calculates the cut cordinates given the fire perimeters geodatabase and a header file and returns a list of the start x, start y, width, and height
>>> auto_coords(fire_num_perim, 'N51117_complex/S2A_MSIL2A_20240729T183921_N0511_R070_T11UMR_20240730T023050_cloudfree.bin_MRAP.hdr')
[2928, 2690, 956, 2246]
'''
from osgeo import gdal
import geopandas as gpd
from pyproj import Proj, Transformer

def extract_map_info_from_gdal(file):
    """
    Extracts map info from a raster file using GDAL.

    Parameters:
    - hdr_file: Path to the raster header file.

    Returns:
    - A tuple with (geotransform, projection)
    """
    # Open the raster file using GDAL
    dataset = gdal.Open(file)
    
    # Extract the geotransform and projection
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    
    return geotransform, projection

def transform_coordinates(x, y, src_proj, tgt_proj):
    """
    Transform coordinates from source projection to target projection.

    Parameters:
    - x: Coordinate in the source projection (easting or longitude)
    - y: Coordinate in the source projection (northing or latitude)
    - src_proj: Source projection as a PROJ string or EPSG code
    - tgt_proj: Target projection as a PROJ string or EPSG code

    Returns:
    - x_transformed, y_transformed: Transformed coordinates in the target projection
    """
    # Define the source and target projections
    proj_src = Proj(src_proj)
    proj_tgt = Proj(tgt_proj)
    
    # Create a transformer for the transformation
    transformer = Transformer.from_proj(proj_src, proj_tgt, always_xy=True)
    
    # Transform coordinates
    x_transformed, y_transformed = transformer.transform(x, y)
    return x_transformed, y_transformed

def gdf_coords(gdf, target_crs):
    """
    Transforms the coordinates of the GeoDataFrame to the specified CRS.

    Parameters:
    - gdf: GeoDataFrame with the fire perimeter data.
    - target_crs: Target CRS as an EPSG code.

    Returns:
    - List of bounding coordinates in the target CRS.
    """
    gdf = gdf.to_crs(epsg=target_crs)
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    return [maxx, maxy, minx, miny]

def auto_coords(fire_num, file, historical_perimeters=None):
    """
    Calculates cut coordinates for raster data and fire perimeters, transforming to a specified projection.

    Parameters:
    - fire_num: List of fire numbers to filter the perimeter data.
    - hdr_file: Path to the raster header file.
    - target_proj: Target projection as a PROJ string or EPSG code

    Returns:
    - List of cut coordinates with a buffer.
    """
    target_proj = 'epsg:3005'
    # Load fire perimeter data
    fire_perims = gpd.read_file('prot_current_fire_polys.shp' if historical_perimeters is None else historical_perimeters)
    
    fire_number_string = 'FIRE_NUMBE' if 'FIRE_NUMBE' in fire_perims else 'FIRE_NUM'
    fire_num_perims = fire_perims[fire_perims[fire_number_string].isin(fire_num)]

    # Extract map info from GDAL
    geotransform, projection = extract_map_info_from_gdal(file)
    # Define the corners using the geotransform
    ulx = geotransform[0]
    uly = geotransform[3]
    pixel_size_x = geotransform[1]
    pixel_size_y = geotransform[5]
    
    # Calculate bottom right corner
    samples = geotransform[1]  # Number of columns
    lines = geotransform[5]    # Number of rows
    brx = ulx - samples * pixel_size_x
    bry = uly + lines * pixel_size_y
    
    # Transform corners to the target projection
    x_top, y_top = transform_coordinates(ulx, uly, projection, target_proj)
    x_pix, y_pix = transform_coordinates(pixel_size_x, pixel_size_y, projection, target_proj)
    x_bot, y_bot = transform_coordinates(brx, bry, projection, target_proj)

    # # Calculate scaling factors
    # print(x_pix,y_pix)
    # x_bot = x_top + x_pix*samples
    # y_bot = y_top - y_pix*lines
    x_con = lines / (x_bot - x_top)
    y_con = samples / (y_bot - y_top)

    # Transform fire perimeter coordinates
    shape_coords = gdf_coords(fire_num_perims, int(target_proj.split(':')[-1]))

    # Calculate cut coordinates
    top_x = (shape_coords[2] - x_top) * x_con
    top_y = (shape_coords[1] - y_top) * y_con
    bot_x = (shape_coords[0] - x_top) * x_con
    bot_y = (shape_coords[3] - y_top) * y_con
    width = bot_x - top_x
    height = bot_y - top_y

    return [int(abs(top_x) - 200), int(abs(top_y) - 200), int(abs(width) + 400), int(abs(height) + 400)]  # Returning cut coordinates with buffer
