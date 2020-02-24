'''
ls *.zip | xargs python3 run.py
'''
# run this in the same file that the zips are in
import os
import sys
args = sys.argv
if len(args) < 2:
    print("Error: run_superres.py [input zip file name]")

for i in range(0, len(args)):
    print("\n")
    fn = args[i]
    if fn.split('.')[-1] != 'zip':
        # only accept zip files
        continue

    if not os.path.exists(fn):
        print("Error: input file does not exist")
        continue

    dn = '.'.join(fn.split('.')[:-1])
    print('mkdir ' + dn)
    cmd = ('python2 ../superres/superres/sentinel2_superres.py ' +
           '--output_file_format ENVI --write_images --num_threads 16 ' +
           fn + ' output.bin')
    print(cmd)
    cmd = 'mv output* ' + dn
    print(cmd)
    cmd = 'mv *.png ' + dn
    print(cmd)
