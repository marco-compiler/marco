#!/bin/bash

git clone https://github.com/marco-compiler/marco-runtime.git
cd marco-runtime
git checkout ${MARCO_RUNTIME_COMMIT}

rm -rf build
mkdir build

cmake \
  -S . -B build -G Ninja \
  -DCMAKE_LINKER_TYPE=MOLD \
  -DCMAKE_BUILD_TYPE=${MARCO_RUNTIME_BUILD_TYPE} \
  -DMARCO_USE_BUILTIN_SUNDIALS=OFF

cmake --build build --target install
rm -rf build
