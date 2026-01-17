#!/bin/bash

MARCO_BUILD_TYPE=${MARCO_BUILD_TYPE:-"Release"}
PYTHON3_EXECUTABLE=${PYTHON3_EXECUTABLE:-"/usr/bin/python3"}

rm -rf marco
git clone https://github.com/marco-compiler/marco.git
cd marco || exit 1
git checkout ${MARCO_COMMIT}

mkdir build

cmake \
  -S . \
  -B build \
  -G Ninja \
  -DCMAKE_LINKER_TYPE=MOLD \
  -DCMAKE_BUILD_TYPE=${MARCO_BUILD_TYPE} \
  -DPython3_EXECUTABLE=${PYTHON3_EXECUTABLE} \
  || exit 1

cmake --build build --target install || exit 1
rm -rf build
