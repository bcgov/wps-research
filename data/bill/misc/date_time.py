'''
Misc file to handle datetime.
'''

import sys
from datetime import datetime, timedelta



def date_str2obj(
        date_str: str,
        format = "%Y%m%d"
):
    '''
    Description
    -----------
    Transforming date in form string to object.

    Currently using format of Sentinel 2 data, YYYY-MM-DD
    '''

    return datetime.strptime(date_str, format).date()



def julian_to_date(year, jday):
    '''
    Converts Julian date (viirs files use this) to our datetime.
    '''
    return datetime(year, 1, 1) + timedelta(days=jday - 1)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python julian_to_date.py <year> <julian_day>")
        sys.exit(1)

    year = int(sys.argv[1])
    jday = int(sys.argv[2])

    dt = julian_to_date(year, jday)
    leap = "yes" if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else "no"

    print(f"{year}-{jday:03d} → {dt.strftime('%Y-%m-%d')} ({dt.strftime('%A')})")
    print(f"Leap year: {leap}")