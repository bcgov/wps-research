'''building SNAP from source
'''

sources = ['snap-engine', 'snap-desktop', 's1tbx', 's2tbx', 's3tbx']

cmds = ['git clone https://github.com/senbox-org/' + x + '.git' for x in sources]

for c in cmds:
    print(c)

print('sudo apt install openjdk-8-jre openjdk-8-jdk')
print('sudo update-java-alternatives --list')
print('sudo update-alternatives --config java')
print('java -version')
print('sudo update-alternatives --config javac')

print("in each toolbox folder:")
print('mvn clean install  -DskipTests=true')

print("how to find out where your java is:")
print("readlink -f /usr/bin/java")

print('add line to .bashrc:')
print('export JAVA_HOME=/path/to/java/bin/')