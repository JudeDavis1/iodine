cmake -B build  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-11
cmake --build build

./build/Iodate/Iodate/Iodate
