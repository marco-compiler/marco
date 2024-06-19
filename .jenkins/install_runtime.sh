#!/bin/bash

git clone https://github.com/marco-compiler/marco-runtime.git
cd marco-runtime
git checkout 58fd4d19c815527bd09a45061f55d208173ab170

rm -rf build
mkdir build

cmake \
  -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=${MARCO_BUILD_TYPE} \
  -DMARCO_USE_BUILTIN_SUNDIALS=OFF

cmake --build build --target install
rm -rf build
