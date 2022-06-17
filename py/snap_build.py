'''building SNAP from source:

How to build SNAP from sources:
https://senbox.atlassian.net/wiki/spaces/SNAP/pages/10879039/How+to+build+SNAP+from+sources

STEP developers: 
https://step.esa.int/main/community/developers/
'''
from misc import run, err
sources = ['snap-engine', 'snap-desktop', 's1tbx', 's2tbx', 's3tbx']
cmds = ['git clone https://github.com/senbox-org/' + x + '.git' for x in sources]

for c in cmds:
    print(c)

# SNAP needs java v. 8! 
print('sudo apt install openjdk-8-jre openjdk-8-jdk')
print('sudo update-java-alternatives --list')
print('sudo update-alternatives --config java')
print('java -version')
print('sudo update-alternatives --config javac')

# https://computingforgeeks.com/how-to-set-default-java-version-on-ubuntu-debian/
print("how to find out where your java is:")
print("readlink -f /usr/bin/java")

print('add line to .bashrc:')
print('export JAVA_HOME=/path/to/java/jre/')a

# assuming your java is set up:
print("in each toolbox folder:")
print('mvn clean install  -DskipTests=true')

print("sudo snap install intellij-idea-ultimate --classic")

# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/24051775/IntelliJ+IDEA
# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/24051775/IntelliJ+IDEA#IntelliJIDEA-RunSNAPDesktopwithadditionalToolboxes(S1%2CS2%2CS3%2C...)


'''
e.g. possible commands:
sudo update-java-alternatives -s java-1.8.0-openjdk-amd64
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64/
export PATH=$PATH:$JAVA_HOME
'''
