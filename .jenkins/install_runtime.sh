#!/bin/bash

git clone https://github.com/marco-compiler/marco-runtime.git
cd marco-runtime
git checkout f7aa2422e563cf10cd5d865be72a442ffde6e871

rm -rf build
mkdir build

cmake \
  -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=${MARCO_BUILD_TYPE} \
  -DMARCO_USE_BUILTIN_SUNDIALS=OFF

cmake --build build --target install
rm -rf build
