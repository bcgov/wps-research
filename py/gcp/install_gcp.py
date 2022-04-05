'''
https://cloud.google.com/sdk/docs/install
'''
import os
import sys
sys.path.append("..")
from misc import run, pd, sep, exists

remote_p  = 'https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/'
remote_f = 'google-cloud-sdk-379.0.0-linux-x86_64.tar.gz'
remote = remote_p + remote_f
dp = os.path.abspath(pd + 'gcp' + sep) + sep  # local directory path
tfn = dp + remote_f  # local target file name

cmd = ' '.join(['wget',
                '--directory-prefix=' + dp,
                'http://dl.google.com/dl/cloudsdk/channels/rapid/downloads/' + remote_f])
print('+r', tfn)
if not exists(tfn):
    run(cmd)

sdk_folder = dp + 'google-cloud-sdk' + sep
if not exists(sdk_folder):
    run('tar xvf ' + tfn)

gsutil = os.popen('which gsutil').read().strip() # what comes up under gsutil
gsutil_t = sdk_folder + 'bin' + sep + 'gsutil' # where our gsutil command should be!
if gsutil != gsutil_t:
    run(sdk_folder + 'install.sh')
    cmd = sep.join(['source ',  'home', os.popen('whoami').read().strip(), '.bashrc'])
    run(cmd)

if gsutil != gsutil_t:
    err('expected: ' + gsutil + ' == ' + gsutil_t)
else:
    print('gsutil installed')

''' do we need this step:?
     ./google-cloud-sdk/bin/gcloud init
'''
