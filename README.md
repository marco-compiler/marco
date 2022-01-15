# MARCO - Modelica Advanced Research COmpiler
MARCO is a prototype compiler for the Modelica language.
It is written in C++ and leverages both the LLVM and MLIR infrastructures.
It also defines an MLIR dialect that is specific to Modelica.

## Requirements
MARCO has been though as a standalone project and thus does not live inside the LLVM repository. This allows for faster configuration and build.
However, the LLVM infrastructures still needs to be installed within the host system and, due to the absence of MLIR in public packages, there is need for a manual build. Moreover, as the MLIR project changes rapidly, the referenced LLVM's commit is periodically updated, in order to be always up-to-date with respect to the LLVM project.

### LLVM
In order to ease the process for newcomers, the instructions needed to build LLVM on the required commit are reported here.

The build type is set to `Release` in order to allow for faster build and execution times, but it strongly suggested setting it to `Debug` when developing on MARCO.
To compile LLVM in a faster way, it is also suggested using the gold linker (`-DLLVM_USE_LINKER=gold`) and enabling the shared libraries (`-DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON`).
Shared libraries also allow for faster MARCO builds.

The install path is optional, but it strongly recommended setting it to a custom path, in order not mix the installed files with system ones or other installations coming from package managers.

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout d7b669b3a30345cfcdb2fde2af6f48aa4b94845d
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install_path -DLLVM_INSTALL_UTILS=True -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;mlir;openmp" ../llvm
make install
```

### OpenModelica
In order to relieve MARCO from the object-oriented characteristics of Modelica, the frontend leverages the OpenModelica frontend to obtain a flattened version of the Modelica sources. The way it is used by MARCO is transparent to the end-user, but yet the compiler must be installed within the system.

The details on how to install OpenModelica are available on its dedicated [website](https://openmodelica.org/).
It is recommended installing the nighly version, because some features leveraged by MARCO may not be present in the stable branch yet.

### Boost
MARCO also needs the Boost libraries to work. They can be retrieved easily through package managers:

```bash
sudo apt install libboost-all-dev
```

### LIT
MARCO uses LLVM's LIT to run the regression tests. It is available through pip, the Python's package manager.

```bash
sudo apt install python3-pip
pip3 install lit
```

## Building MARCO
Once all the requirements have been fulfilled, MARCO can be built.
However, note that the repository contains submodules which must be initialized first as follows:

```bash
git submodule update --recursive --init
```

The project can be built with the following instructions.
The `LLVM_PATH` variable must be set to the installation path that was used during the LLVM installation process.
The `LLVM_EXTERNAL_LIT` variable represent the path (including the executable name) to the `lit` tool. If it has been installed in user mode, it is usually `/home/user/.local/bin/lit`.

```
cd marco
mkdir build && cd build
cmake -DLLVM_PATH=llvm_install_path -DLLVM_EXTERNAL_LIT=lit_executable_path -DCMAKE_BUILD_TYPE=Release ..
make all
```

Doxygen documentation can also be built by adding the `-DMARCO_BUILD_DOCS=ON` option. Files will be installed inside the `docs` folder.

## Tests
MARCO adopts a test infrastructure that is similar to the one of LLVM, thus defining two categories of tests: unit tests and regression tests.

Unit tests are written using [Google Test](https://github.com/google/googletest/) and Google Mock and are located in the `unittest` directory.
Regression tests, instead, leverage the LIT tool mentioned earlier, and are located in the `test` directory.

In general, unit tests are reserved to ensure the correct behaviour of the data structures, while regression tests are preferred to test broader scopes such as transformation or even the whole compilation pipeline. 

Unit tests can be run with `make test`, while regression tests can be run with `make check`.
