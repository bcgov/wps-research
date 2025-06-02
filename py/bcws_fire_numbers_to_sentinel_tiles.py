'''20250602: bcws_fire_numbers_to_s2_tiles.py: based on a code by Sterling von Dehn.
Checks which tiles are required to view a fire using sentinel 2 data. Takes a fire number and returns a list of tiles
Assumes the perimeters / points files stored in this folder are current 
>>> check_tile_id('G90267')
>>> check_tile_id(['N51117','N51069', 'N51210','N51103','N51228'])
'''
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from misc import run
import os


tile_path = 'sentinel2_bc_tiles_shp/Sentinel_BC_Tiles.shp'

def check_tile_id(fire_num, historical_perimeters=None):
    '''
    Checks which tiles a fire numbers perimeter is in, for downloading
    Can take a single fire number or a list of fire numbers
    '''

    #if historical_perimeters is None:
    #    retcode = run('python3 py/get_perimeters.py')

    fire_perims_path = 'prot_current_fire_polys.shp' if historical_perimeters is None else historical_perimeters
    #checking if multiple fire numbers are given
    if type(fire_num) == str:
        fire_num = [fire_num]

    #reading files
    fire_perims = gpd.read_file(fire_perims_path)
    fire_perims = fire_perims.to_crs(epsg=4326)
    fire_number_string = 'FIRE_NUM' if 'FIRE_NUM' in fire_perims else 'FIRE_NUMBE'
    fire_num_perim = fire_perims[fire_perims[fire_number_string].isin(fire_num)]
    tile_id = gpd.read_file(tile_path)
    tile_id = tile_id.to_crs(epsg=4326)

    tile_names = []

    # Create a figure and axis for plotting
    fig, ax = plt.subplots()

    # Define colors for plotting
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    color_index = 0

    # list the tiles that intersect the various fires
    containing = gpd.GeoDataFrame(columns=tile_id.columns, geometry='geometry', crs=tile_id.crs)
    
    # Loop over each fire perimeter feature
    for _, fire_geom in fire_num_perim.iterrows():
        fire_geometry = fire_geom.geometry
        # Check which tiles intersect with the current fire perimeter
        intersecting = tile_id[tile_id.geometry.intersects(fire_geometry)]

        # Accumulate all intersecting tiles so far
        containing = pd.concat([containing, intersecting])
        print(containing)

        # Save tile names for the current fire perimeter
        for name in intersecting['Name']:
            if pd.notna(name) and f'T{name}' not in tile_names:  # Check if the name is not NaN
                tile_names.append(f'T{name}')

    # Drop duplicates from containing
    containing = containing.drop_duplicates(subset='Name')

    # Plot the tiles
    sections = []
    for idx, section in containing.iterrows():
        if section["Name"] not in sections:
            tile_id.loc[[idx]].plot(ax=ax, edgecolor='black', color=colors[color_index])
            ax.scatter(np.nan, np.nan, color=colors[color_index], marker='s', s=60, label=f'T{section["Name"]}')
            color_index = (color_index + 1) % len(colors)  # Cycle through colors
            sections.append(section["Name"])

    # Plot the fire perimeter
    fire_num_perim.plot(ax=ax, edgecolor='black', color='red')

    # Set plot title and legend
    ax.set_title(f'Fire Perimeters in Sentinel2 Tiles', fontsize=14)
    ax.legend(fontsize=14)

    # Show plot
    plt.show()

    # Return the list of tile names
    return tile_names

if __name__ == "__main__":
    import sys
    args = sys.argv
    fire_num = args[1:]
    print(check_tile_id(fire_num))

