'''
donwloads sentinel data, extracts cloudfree swir nir bands, chooses most recent avaiable pixels, and merges frames (if necessary). Automatic date range is defined if no end date is given. Start download date is always 3 weeks prior to (first) fire start date (in case of complex of fires specified).
$python3 get_composite.py G90267 #single fire with automatic end date
$python3 get_composite.py 20240630 G90267 #single fire with manual end date
$python3 get_composite.py N51117 N51069 N51210 N51103 #fire complex with automatic end date
$python3 get_composite.py 20240810 N51117 N51069 N51210 N51103 #fire complex with manual end date

20250605 should add feature to guess the name of the fire ignition field : ) 
'''
from misc import run, args, extract_date, exist, err
from percent_vs_time import extract_data_percent
from cut_coords import plot_image_with_rectangle
from barc_comp import trim_tif_to_shapefile
from datetime import datetime, timedelta
from check_tile_id import check_tile_id
from auto_coords import auto_coords
from dnbr import barc_time_series
import geopandas as gpd
from plot import plot
import os

no_update_listing = False
skip_download = False
historical_perimeters = None  # for retroactive / historical analysis mode.. fire perimters stored offline in shapefile`
historical_points = None

def is_valid_date(date_string):
    '''
    Check if string is 8 digits long & consists of digits parseable into datetime object
    '''
    if len(date_string) != 8 or not date_string.isdigit():
        return False
    try:   # try to parse string as date in yyyymmdd format
        datetime.strptime(date_string, '%Y%m%d')
        return True
    except Exception as e:
        return False


def get_composite_image(fire_num, end_date=None):
    global no_update_listing, skip_download
    '''
    Takes a fire number as well as a tile ID and downloads an MRAP timesires composite
    '''
    #changing single fire numbers to list and naming multi fire scene
    if len(fire_num) > 1:
        fire_name = f'{fire_num[0]}_complex'
    else:
        fire_name = fire_num[0]
    
    #taking end date as today none defined
    if end_date == None:
        end_date = datetime.today().date()
        str_end_date_comps = str(end_date).split('-')
        str_end_date = ''
        for comp in str_end_date_comps:
            str_end_date += comp
    else:
        str_end_date = end_date
    
    #getting the ignition date for fires and taking the smallest.. for historical case we can revert to polygon file
    fire_points_path = 'prot_current_fire_points.shp' if historical_perimeters is None else historical_perimeters  # add historical data option : ) 
    fire_points = gpd.read_file(fire_points_path)
    fire_points = fire_points.to_crs(epsg=4326)

    fire_number_string = 'FIRE_NUM' if 'FIRE_NUM' in fire_points else 'FIRE_NUMBE'
    fire_num_point = fire_points[fire_points[fire_number_string].isin(fire_num)]
    
    ignt_dates = None
    # ignition date field name might not be normalised
    try:
        ignt_dates = [datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S').date() for date in  fire_num_point.IGNITN_DT]
    except:
        try:
            ignt_dates = [datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S').date() for date in  fire_num_point.FIRE_DATE]
        except:
            try:
                ignt_dates = [datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S').date() for date in  fire_num_point.IGNITION_D]
            except:
                err("could not parse fire start date from shapefile")

    fire_start_date = min(ignt_dates)
    start_date = fire_start_date - timedelta(weeks=3)

    #changing date format to string for sync_daterange script
    str_start_date_comps = str(start_date).split('-')
    str_start_date = ''
    for comp in str_start_date_comps:
        str_start_date += comp

    #str_start_date = '20230201' # FOR DONNIE COMPLEX REMOVE FOR OTHER USE!!!!!!!!!!!!!!!!!!
    # historical perimeters may not include end date? NEED TO WORK ON THIS
    if historical_perimeters:
        str_end_date = str_start_date[:4] + '1111'

    tiles = check_tile_id(fire_num, historical_perimeters) #checking tiles
    tile_str = ''
    for tile in tiles:
        if not os.path.exists(f'L2_{tile}/*{str_end_date}*'):
            tile_str += f' {tile}'
    
    if tile_str != '' and not skip_download:
        # defining sync string
        sync_string = f'python3 py/sync_daterange_gid_zip.py {str_start_date} {str_end_date}' + \
                        tile_str + (' --no_update_listing' if no_update_listing else '')
        run(sync_string) #running download script
    
    run('python3 py/sentinel2_extract_cloudfree_swir_nir.py') #running cloudfree extraction
    for tile in tiles:
        run(f'python3 py/sentinel2_mrap.py L2_{tile}') #running MRAP script
    
    if len(tiles) > 1:
        run(f'python3 py/sentinel2_mrap_merge.py {fire_name} {tile_str}') #running merge script if necesary 

    #Making fire directory and copying MRAP frames in
    else:
        if not os.path.exists(fire_name):
            os.mkdir(fire_name)
        for tile in tiles:
            run(f'cp L2_{tile}/*MRAP* {fire_name}')
    
    #getting list of files for cutting
    files = [x.strip() for x in os.popen(f'ls -1 {fire_name}/*.bin').readlines()] 
    files.sort()

    #cut_data = plot_image_with_rectangle(files[-1]) #prompt user for cut coords
    cut_data = auto_coords(fire_num,
                           files[-1],
                           historical_perimeters)  # 20241211 add option for historical fires/perimeters
    
    run(f'python3 py/cut.py {fire_name} {int(cut_data[0])} {int(cut_data[1])} {int(cut_data[2])} {int(cut_data[3])}')

    if len(tiles) == 1:
        run(f'rm -r {fire_name}')

    #reading through files to find nearest date to start date
    files = os.listdir(f'{fire_name}_cut')
    date_list = []
    for n in range(len(files)):
        if files[n].split('.')[-1] == 'bin':
            date_list.append(extract_date(files[n]))
        else:
            continue

    barc_start = None
    date_list = sorted(date_list)
    print("date_list", date_list)
    for i in range(len(date_list)):
        if datetime.strptime(date_list[i], '%Y%m%d').date() >= (fire_start_date - timedelta(days=1)):
            barc_start = date_list[i-1]
            break

    print("barc_start", barc_start)
    extract_data_percent(f'{fire_name}_cut',
                         barc_start) #plotting the data percent vs time for frames

    # #plotting image, NBR, dNBR time series
    plot(f'{fire_name}_cut',
         fire_name)  # places files into three directories: 'images', 'NBR', and 'dNBR'
    
    # plotting BARC time series
    barc_time_series(f'{fire_name}_cut',
                     int(barc_start),
                     f'{fire_name}')
    
    for fire in fire_num:
        barc_files = [x.strip() for x in os.popen('ls -1 ' + f'{fire}_barcs/*BARC.tif').readlines()]
        for b in barc_files:
            # trim tif to shapefile ( recorded fire perimeter ) 
            trim_tif_to_shapefile(b,
                                  fire_name,
                                  '.'.join(b.split('.')[:-1]) + '_clipped.tif',
                                  historical_perimeters)


if __name__ == "__main__":
    end_date = args[1] if is_valid_date(args[1]) else None  # date arg possibly at position 1

    no_update_listing, skip_download = "--no_update_listing" in args, "--skip_download" in args  # nonpositional options

    for i in args[1:]:
        if i[:2] == '--':
            w = i.split('=')
            if w[0] == '--historical_perimeters':
                print("historical mode")
                historical_perimeters = w[1]
                if not exist(historical_perimeters):
                    err("could not find file: " + str(historical_perimeters))

            '''            
            if w[0] == '--historical_points':
                historical_points = w[1]
                if not exist(historical_points):
                    err("could not find file: " + str(historical_points))
            '''
    if historical_perimeters is None:
        return_code = os.system('python3 py/get_perimeters.py')  # refresh perimeters, n.b. should add "past data" option

    fire_numbers = []  # other options assumed to be fire ID codes
    for i in args[1:]:
        if (i[0:2] != '--') and (not is_valid_date(i)):
            fire_numbers += [i]
    
    get_composite_image(fire_numbers, end_date)  # get_composite_image() is in this file!
