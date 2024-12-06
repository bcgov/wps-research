'''
Prerequisite: 
To install EODMS cli:

python3 -m pip install eodms-api-client
'''

import os
os.system('eodms -c RCM -s 2021-11-15 -e 2021-12-01 -g abbotsford.json --dump-results')
