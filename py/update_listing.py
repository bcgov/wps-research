'''20240621 update listing of objects in nrcan sentinel products mirror
'''
from misc import timestamp, exists, sep
import os

my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep

def update_listing():
    ts = timestamp()
    cmd = ' '.join(['aws',  # read data from aws
                    's3api',
                    'list-objects',
                    '--no-sign-request',
                    '--bucket sentinel-products-ca-mirror'])
    print(cmd)
    data = os.popen(cmd).read()

    if not exists(my_path + 'listing'):  # json backup for analysis
        os.mkdir(my_path + 'listing')

    df = my_path + 'listing' + sep + ts + '_objects.txt'  # file to write
    print('+w', df)
    open(df, 'wb').write(data.encode())  # record json to file
    print('done')

def latest_listing():
    listings = os.popen('ls -1 '  + my_path + 'listing' + sep + '*_objects.txt').readlines()
    listings = [[x.split(sep)[-1].split('_')[0], x.strip()] for x in listings]
    listings.sort()  # sort on increasing time
    latest = listings[-1][1]  #  most recent entry
    print('+r', latest)
    return open(latest).read()

if __name__ == '__main__':
    update_listing()
    d = latest_listing()
