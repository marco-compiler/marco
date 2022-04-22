mkdir build
cd build

git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout d7b669b3a30345cfcdb2fde2af6f48aa4b94845d
mkdir build
cd build

:: Disable openmp as it is not supported by msbuild
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=C:\llvm -DLLVM_INSTALL_UTILS=True -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;mlir" ../llvm
msbuild .\INSTALL.vcxproj /p:Configuration=Release
cd ../../..