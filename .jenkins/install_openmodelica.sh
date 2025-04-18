#!/bin/bash

rm -rf OpenModelica
git clone https://github.com/OpenModelica/OpenModelica.git
cd OpenModelica || exit 1
git checkout ${OPENMODELICA_COMMIT}
git submodule update --force --init --recursive

rm -rf build
mkdir build

cmake \
  -S . \
  -B build \
  -G Ninja \
  -DCMAKE_LINKER_TYPE=MOLD \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DOM_USE_CCACHE=OFF \
  -DOM_ENABLE_GUI_CLIENTS=OFF \
  || exit 1

cmake --build build --target install || exit 1
rm -rf build
