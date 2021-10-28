# tested on Ubuntu 20
tar xvf jpy-0.9.0.tar.gz
sudo apt install openjdk-17-jre
sudo apt install openjdk-17-jdk

# might need to insert these in .bashrc
export JDK_HOME=/usr/lib/jvm/java-17-openjdk-amd64/bin/
export JAVA_HOME=$JDK_HOME
python3 get-pip.py
python3 setup.py --maven bdist_wheel
