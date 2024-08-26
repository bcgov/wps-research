'''20240826 (last test date) building SNAP from source:

1) Note: assume SNAP repos will be cloned into the PRESENT working folder. WARNING: need to rename snap executable to (e.g.):
esa-snap

to make sure "snap" package manager and ESA SNAP software don't collide (i.e. your computer tries to run SNAP every time the OS tries to install new software..

Background: 
* How to build SNAP from sources: https://senbox.atlassian.net/wiki/spaces/SNAP/pages/10879039/How+to+build+SNAP+from+sources
* STEP developers:  https://step.esa.int/main/community/developers/
'''
from misc import run, err
import sys
import os

sources = ['snap-engine', 'snap-desktop', 's1tbx', 's2tbx', 's3tbx']
cmds = [[x, 'git clone https://github.com/senbox-org/' + x + '.git'] for x in sources]

for x, c in cmds:
    if not os.path.exists(x + os.path.sep + '.git'):
        run(c)
    else:
        print('+r ' + x + '.git')

# SNAP needs java v. 8! 
lines = [x.strip().split('/')[0] for x in os.popen('apt list --installed | grep jdk').readlines()]
to_install = ['openjdk-11-jre', 'openjdk-11-jdk', 'maven']
print(lines)
for x in to_install:
    if x not in lines:
        run('sudo apt install ' + x)

# install intellij-idea-ultimate
run("sudo snap install intellij-idea-ultimate --classic")

# not sure if we need these commands
a = os.system('sudo update-java-alternatives --list')
a = os.system('sudo update-alternatives --config java')
a = os.system('java -version')
a = os.system('sudo update-alternatives --config javac')


# where is java ?
# java_path = os.popen('readlink -f /usr/bin/javac').read().strip()  # where is your java?
# print(java_path)

java_path = os.popen('sudo update-java-alternatives --list').readlines()[-1].split()[-1]

bashrc = '/home/' + os.popen('whoami').read().strip() + os.path.sep + '.bashrc'
lines = [x.rstrip() for x in open(bashrc).readlines()]  # existing bashrc lines
new_lines = []
for line in lines:
    if len(line.split('JAVA_HOME')) > 1:
        pass
    else:
        new_lines += [line]
new_lines += ['\nexport JAVA_HOME=' + java_path + ' # setting for ESA SNAP software']  # put this at end

print('+w', bashrc)
open(bashrc, 'wb').write(('\n'.join(new_lines)).encode())

# assuming your java is set up:
# print("in each toolbox folder:")

source = ('source /home/' + os.popen('whoami').read().strip() + '/.bashrc')
print(c)

# build the repos in parallel
for s in sources:
    c = 'cd ' + s + '; mvn clean install  -DskipTests=true & '
    a = os.system(c)

# but now how do we run the SNAP we just built?

# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/24051775/IntelliJ+IDEA
# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/24051775/IntelliJ+IDEA#IntelliJIDEA-RunSNAPDesktopwithadditionalToolboxes(S1%2CS2%2CS3%2C...)


'''
e.g. possible commands:
sudo update-java-alternatives -s java-1.8.0-openjdk-amd64
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64/
export PATH=$PATH:$JAVA_HOME
'''
