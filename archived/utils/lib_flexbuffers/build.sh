set -e

echo "****************************"
echo "* Building lib_flexbuffers"
echo "****************************"

mkdir -p build
cd build
cmake ../
make install
cd ..
