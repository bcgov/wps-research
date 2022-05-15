cd ~/GitHub/
mkdir -p snap


<<comment
cd snap
git clone https://github.com/senbox-org/snap-engine.git
cd snap-engine
mvn clean install
  
cd ..
git clone https://github.com/senbox-org/snap-desktop.git
cd snap-desktop
mvn clean install

cd snap
cd snap-engine
mvn clean install

cd ..
cd snap-desktop
mvn clean install

comment
cd snap
git clone https://github.com/senbox-org/s1tbx.git
cd s1tbx
mvn clean install
cd ..
cd ..

cd snap
git clone https://github.com/senbox-org/s2tbx.git
cd s2tbx
mvn clean install
cd .. 
cd ..
