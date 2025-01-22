#!/bin/bash

rm -rf llvm-project
git clone https://github.com/marco-compiler/llvm-project.git
cd llvm-project || exit 1
git checkout ${LLVM_COMMIT}

mkdir build

cmake \
  -S llvm \
  -B build \
  -G Ninja \
  -DCMAKE_LINKER_TYPE=MOLD \
  -DCMAKE_BUILD_TYPE=${LLVM_BUILD_TYPE} \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_INSTALL_UTILS=True \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir;openmp" \
  -DLLVM_ENABLE_ASSERTIONS=${LLVM_ENABLE_ASSERTIONS} \
  -DLLVM_PARALLEL_COMPILE_JOBS=${LLVM_PARALLEL_COMPILE_JOBS} \
  -DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS} \
  || exit 1

cmake --build build --target install || exit 1
rm -rf build
