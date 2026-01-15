








if __name__ == "__main__":

    from misc.sen2 import get_date_dict
    from raster import (
        Raster,
        minimum_nan_raster
    )


    #Get the unique dates in ascending order, just dates
    unique_dates = get_date_dict("./fire_C11659")


    #Load the raster, can take some time to filter rasters
    #Can be memory heavy, for testing use a folder with just a few ENVI files.
    raster_by_date_dict = {}

    for date, files in unique_dates.items():

        date_str = date.isoformat()

        raster, _ = minimum_nan_raster(files)

        raster_by_date_dict[date_str] = raster