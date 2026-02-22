'''
Date ordering
'''

from datetime import datetime, timedelta

from raster import Raster


def extract_datetime(
        filename: str
) -> datetime:
    
    '''
    datetime in ISO 8601 in UTC
    '''
    
    raster = Raster(filename)

    try:
        acquisition_time = raster.meta['acquisition_time']
        return datetime.fromisoformat(acquisition_time).replace(tzinfo=None)

    except Exception:
        raise ValueError(f"Cannot extract acquisition timestamp from \n{filename}")
    


def get_ordered_file_dict(
        image_dir: str,
        mask_dir: str = None,
        *,
        start: datetime = None,
        end: datetime = None
):

    from misc import iter_files

    dictionary = {}

    #1. Iterate through files and get names
    for img_f in iter_files(image_dir, '.bin'):
        acquisition_time = extract_datetime(img_f)
        
        if (start is not None and acquisition_time < start) or \
            (end is not None and acquisition_time > end):

            continue

        #Filters only necessary dates.
        dictionary[acquisition_time] = {'image_path': img_f}

    dictionary = dict(sorted(dictionary.items()))

    #2. Iterate through mask files and save file
    ommited_mask_f = 0

    for mask_f in iter_files(mask_dir, '.bin'):

        acquisition_time = extract_datetime(mask_f)

        try:
            dictionary[acquisition_time]['mask_path'] = mask_f

        except KeyError:
            ommited_mask_f += 1

    print(f'Iterating completed | ommited {ommited_mask_f} mask files in total.')
    
    #3. Last inspection 

    missing_dates = [
        date for date, file_dict in dictionary.items()
        if 'mask_path' not in file_dict
    ]

    if missing_dates:
        print("Removing entries without mask_path:")
        for d in missing_dates:
            print(f" - {d}")

    dictionary = {
        date: file_dict
        for date, file_dict in dictionary.items()
        if 'mask_path' in file_dict
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








    




