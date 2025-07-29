#!/bin/bash

cd marco

cmake \
  -S . \
  -B build \
  -G Ninja \
  -DCMAKE_LINKER_TYPE=MOLD \
  -DCMAKE_BUILD_TYPE=${MARCO_BUILD_TYPE} \
  -DPython3_EXECUTABLE=${PYTHON3_EXECUTABLE} \
  || exit 1

cmake --build build --target install || exit 1