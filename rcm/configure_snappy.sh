# tested on Ubuntu 20 with ESA SNAP 8.0 which demands java 8
# configure snappy by installing java 8 stuff, to build:
#     * jpy (python/java bridge)
# finally then showing snappy where python is!

tar xvf jpy-0.9.0.tar.gz
sudo apt install openjdk-8-jre openjdk-8-jdk python3-pip maven

# might need to insert these in .bashrc
export JDK_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export JAVA_HOME=$JDK_HOME


python3 -m pip install wheel
cd jpy-0.9.0
python3 setup.py --maven bdist_wheel

cp -v dist/*.whl "/home/$USER/.snap/snap-python/snappy"

# configure snappy
/usr/local/snap/bin/snappy-conf /usr/bin/python3

# note, the above command doesn't complete for some reason.. 
# might have to execute this separately, don't know why
cp -rv "/home/$USER/.snap/snap-python/snappy/" "/home/$USER/.local/lib/python3.8/site-packages/"

