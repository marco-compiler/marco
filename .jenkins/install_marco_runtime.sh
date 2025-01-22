#!/bin/bash

rm -rf marco-runtime
git clone https://github.com/marco-compiler/marco-runtime.git
cd marco-runtime || exit 1
git checkout ${MARCO_RUNTIME_COMMIT}

rm -rf build
mkdir build

cmake \
  -S . -B build -G Ninja \
  -DCMAKE_LINKER_TYPE=MOLD \
  -DCMAKE_BUILD_TYPE=${MARCO_RUNTIME_BUILD_TYPE} \
  -DMARCO_USE_BUILTIN_SUNDIALS=OFF \
  || exit 1

cmake --build build --target install || exit 1
rm -rf build
