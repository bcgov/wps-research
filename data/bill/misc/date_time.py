'''
Misc to handle datetime for faster and easier use.
'''


from datetime import datetime

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