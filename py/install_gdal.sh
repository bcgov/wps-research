# 20230430 install latest GDAL by building from source
wget https://github.com/OSGeo/gdal/releases/download/v3.6.4/gdal-3.6.4.tar.gz
tar xvf gdal-3.6.4.tar.gz
# git clone https://github.com/OSGeo/gdal.git
# cd gdal
cd gdal-3.6.4
mkdir -p build
cd build
cmake ..
# cmake ..  -DBUILD_PYTHON_BINDINGS:ON
cmake --build .
sudo cmake --build . --target install

# https://github.com/OSGeo/gdal/releases/download/v3.6.4/gdal-3.6.4.tar.gz
pip3 uninstall gdal
pip3 install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
