#!/bin/bash

git clone https://github.com/OpenModelica/OpenModelica.git
cd OpenModelica
git checkout e499b206639233a2e3fb5138bacd9513c500de6c
git submodule update --force --init --recursive

rm -rf build
mkdir build

cmake \
  -S . \
  -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DOM_USE_CCACHE=OFF \
  -DOM_ENABLE_GUI_CLIENTS=OFF

cmake --build build --target install
rm -rf build
