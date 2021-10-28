# tested on Ubuntu 20 with ESA SNAP 8.0 which demands java 8
tar xvf jpy-0.9.0.tar.gz
sudo apt install openjdk-8-jre openjdk-8-jdk python3-pip maven

# might need to insert these in .bashrc
export JDK_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export JAVA_HOME=$JDK_HOME

cd jpy-0.9.0
python3 setup.py --maven bdist_wheel

