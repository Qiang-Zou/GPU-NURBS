cd ./src
rm -rf build
mkdir build
cd build
cmake ..
echo "make -j$(nproc)"
make -j$(nproc)
