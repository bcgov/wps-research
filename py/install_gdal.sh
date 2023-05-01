# 20230430 install latest GDAL by building from source
git clone https://github.com/OSGeo/gdal.git
cd gdal
mkdir -p build
cd build
# cmake ..
cmake ..  -BUILD_PYTHON_BINDINGS:ON
cmake --build .
sudo cmake --build . --target install
