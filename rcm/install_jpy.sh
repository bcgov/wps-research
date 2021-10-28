# tested on Ubuntu 20
tar xvf jpy-0.9.0.tar.gz
sudo apt install openjdk-17-jre openjdk-17-jdk python3-pip

# might need to insert these in .bashrc
export JDK_HOME=/usr/lib/jvm/java-17-openjdk-amd64/bin/
export JAVA_HOME=$JDK_HOME

cd jpy-0.9.0
python3 setup.py --maven bdist_wheel
