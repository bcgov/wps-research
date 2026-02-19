'''
Date ordering
'''

from datetime import datetime, timedelta

from raster import Raster


def extract_datetime(
        filename: str
) -> datetime:
    
    '''
    Returns
    -------
        datetime in ISO 8601 in UTC
    '''
    
    raster = Raster(filename)

    try:
        acquisition_time = raster.meta['acquisition_time']
        return datetime.fromisoformat(acquisition_time)

    except Exception:
        raise ValueError(f"Cannot extract acquisition timestamp from \n{filename}")
    


def get_ordered_file_dict(
        image_dir: str,
        cloud_dir: str = None
):
    '''
    Description
    -----------
    '''
    from misc import iter_files

    dictionary = {}

    #1. Iterate through files and get names

    for img_f in iter_files(image_dir, '.bin'):
        acquision_time = extract_datetime(img_f)
        dictionary[acquision_time] = {'image_path': img_f}

    dictionary = dict(sorted(dictionary.items()))

    #2. Iterate through cloud files and save file
    ommited_cloud_f = 0

    for cloud_f in iter_files(cloud_dir, '.bin'):

        acquision_time = extract_datetime(cloud_f)

        try:
            dictionary[acquision_time]['cloud_path'] = cloud_f

        except KeyError:
            ommited_cloud_f += 1
            # print(f'Ommitted cloud: No file of timestamp = {acquision_time} in image dir')

    print(f'Iterating completed | ommited {ommited_cloud_f} cloud files in total.')
    
    #3. Last inspection 

    missing_dates = [
        date for date, file_dict in dictionary.items()
        if 'cloud_path' not in file_dict
    ]

    if missing_dates:
        print("Removing entries without cloud_path:")
        for d in missing_dates:
            print(f" - {d}")

    dictionary = {
        date: file_dict
        for date, file_dict in dictionary.items()
        if 'cloud_path' in file_dict
    }

    print(f"Dictionary completed | it stores {len(dictionary)} timestamps in total.")

    return dictionary
    


def get_dates_within(
        datetime_list: list[datetime],
        current_datetime: datetime,
        N_days: int
):
    '''
    Get all dates that are within N days before the current date.
    '''

    lower_bound = current_datetime - timedelta(days=N_days)

    for i in range(len(datetime_list)):

        if datetime_list[i] >= lower_bound:

            return datetime_list[i:]
        
    return []



if __name__ == "__main__":

    dictionary = get_ordered_file_dict(
        image_dir='C11659/L1C/extracted',
        cloud_dir='C11659/cloud_60m'
    )
    
    print(dictionary)








    




