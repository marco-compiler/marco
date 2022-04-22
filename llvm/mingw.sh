#!/bin/bash

# Download repo of PKGBUILDS for mingw64, checkout version 13 of llvm PKGBUILD
git clone https://github.com/msys2/MINGW-packages.git
cd MINGW-packages
git checkout b845c4cc40a43c60558f0df1d4f3aed6b930fcf3
mv mingw-w64-clang/ ../build
cd ..
rm -rf MINGW-packages

# Run llvm PKGBUILD patching it to compile mlir beforehand
cp marco.patch build
cd build
patch PKGBUILD < llvm-mlir-mingw.patch
makepkg -Csi --skipinteg --noconfirm --nocheck
cd ..