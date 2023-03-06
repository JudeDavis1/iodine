cmake -B build  -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-11
cmake --build build

clear
echo Beginning execution:
echo 
echo

./build/Iodate/Iodate/Iodate
