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
        dt = datetime.fromisoformat(acquisition_time.replace("Z", ""))
        return dt.replace(microsecond=0, tzinfo=None)

    except Exception:
        raise ValueError(f"Cannot extract acquisition timestamp from \n{filename}")
    


def get_ordered_file_dict(
        image_dir,
        mask_dir=None,
        *,
        start: datetime = None,
        end: datetime = None
):
    
    from misc import iter_files

    dictionary = {}

    # Normalize to list
    image_dirs = [image_dir] if isinstance(image_dir, str) else list(image_dir)
    
    # Collect image files
    for i_dir in image_dirs:
        for img_f in iter_files(i_dir, '.bin'):
            acquisition_time = extract_datetime(img_f)
            if (start is not None and acquisition_time < start) or \
               (end is not None and acquisition_time > end):
                continue
            dictionary[acquisition_time] = dictionary.get(acquisition_time, {})
            dictionary[acquisition_time].setdefault('image_path', []).append(img_f)

    # Remove entries without all image dirs represented
    dictionary = {
        date: file_dict
        for date, file_dict in dictionary.items()
        if len(file_dict.get('image_path', [])) == len(image_dirs)
    }

    dictionary = dict(sorted(dictionary.items()))

    # If no mask_dir → return early
    if mask_dir is None:
        print(f"Dictionary completed | it stores {len(dictionary)} timestamps.")
        return dictionary

    # Normalize to list
    mask_dirs = [mask_dir] if isinstance(mask_dir, str) else list(mask_dir)

    # Collect mask files
    omitted_mask_f = 0
    for m_dir in mask_dirs:
        for mask_f in iter_files(m_dir, '.bin'):
            acquisition_time = extract_datetime(mask_f)
            if acquisition_time in dictionary:
                dictionary[acquisition_time].setdefault('mask_path', []).append(mask_f)
            else:
                omitted_mask_f += 1
    print(f'Iterating completed | omitted {omitted_mask_f} mask files.')

    # Remove entries without all mask dirs represented
    dictionary = {
        date: file_dict
        for date, file_dict in dictionary.items()
        if len(file_dict.get('mask_path', [])) == len(mask_dirs)
    }

    print(f"Dictionary completed | it stores {len(dictionary)} timestamps.")
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
        
    return [d for d in datetime_list if lower_bound <= d < current_datetime]








    




