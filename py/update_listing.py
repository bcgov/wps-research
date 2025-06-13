'''20240621 update listing of objects in nrcan sentinel products mirror
'''
from misc import timestamp, exists, sep
import os

my_path_0 = '/data/'
my_path_1 = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
my_path = my_path_0

try:  # json backup
    if not exists(my_path_0):
        print('mkdir', my_path_0)
        os.mkdir(my_path_0)
except:
    if not exists(my_path_1 + '.listing'): 
        os.mkdir(my_path_1 + '.listing')

def update_listing():
    global my_path
    if exists(my_path_0):
        my_path = my_path_0
    else:
        my_path = my_path_1
    if not exists(my_path):
        err('path not found: ' + str(my_path))

    ts = timestamp()
    cmd = ' '.join(['aws',  # read data from aws
                    's3api',
                    'list-objects',
                    '--no-sign-request',
                    '--bucket sentinel-products-ca-mirror'])
    print(cmd)
    data = os.popen(cmd).read()
    
    df = my_path + '.listing' + sep + ts + '_objects.txt'  # file to write
    print('+w', df)
    open(df, 'wb').write(data.encode())  # record json to file
    print('done')

def remove_empty_files(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and os.path.getsize(filepath) == 0:
            os.remove(filepath)
            print(f"Deleted: {filepath}")

def latest_listing():
    listings = os.popen('ls -1 '  + my_path + '.listing' + sep + '*_objects.txt').readlines()
    listings = [[x.split(sep)[-1].split('_')[0], x.strip()] for x in listings]
    listings.sort()  # sort on increasing time
    latest = listings[-1][1]  #  most recent entry
    print('+r', latest)
    return open(latest).read()

if __name__ == '__main__':
    print("removing empty listings..")
    remove_empty_files( my_path + '.listing' )
    update_listing()
    d = latest_listing()
