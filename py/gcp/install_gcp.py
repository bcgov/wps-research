'''https://cloud.google.com/sdk/docs/install'''
import os
import sys
sys.path.append("..")
from misc import run, pd, sep, exists

remote_p  = 'https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/'
remote_f = 'google-cloud-sdk-379.0.0-linux-x86_64.tar.gz'
remote = remote_p + remote_f # remote path

dp = os.path.abspath(pd + 'gcp' + sep) + sep  # local directory path
tfn = dp + remote_f  # local target file name

cmd = ' '.join(['wget',
                '--directory-prefix=' + dp,
                'http://dl.google.com/dl/cloudsdk/channels/rapid/downloads/' + remote_f])
print('+r', tfn)
if not exists(tfn):
    run(cmd)  # download the remote file

sdk_folder = dp + 'google-cloud-sdk' + sep  # local extraction location
if not exists(sdk_folder):
    run('tar xvf ' + tfn)

gcloud_bin = os.popen('which gcloud').read().strip() # what comes up under gcloud?
gcloud_t = sdk_folder + 'bin' + sep + 'gcloud' # where our gcloud command should be!
if gcloud_bin != gcloud_t:
    run(sdk_folder + 'install.sh')  # add install info to ~/.bashrc file
    cmd = sep.join(['source ',  'home', os.popen('whoami').read().strip(), '.bashrc'])
    run(cmd) # load the ~.bashrc file to be able to use the command

if gcloud_bin != gcloud_t:  # check command without prefix in expected location
    err('expected: ' + gcloud_bin + ' == ' + gcloud_t)
else:
    print('gcloud installed')

''' do we need this step:?
     ./google-cloud-sdk/bin/gcloud init
'''

# install crcmod
try:
    import crcmod
except Exception:
    run('sudo apt-get install gcc python3-dev python3-setuptools')
    run('sudo pip3 uninstall crcmod')
    run('sudo pip3 install --no-cache-dir -U crcmod')
