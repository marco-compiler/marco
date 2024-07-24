# Building on Linux / macOS
We provide here directions to install MARCO on Linux and macOS.
The commands for installing some packages are based on the `apt` package manager, but it can be easily adapted to others.

## Requirements
### LLVM
MARCO has been though as a standalone project and thus does not live inside the LLVM repository.
This allows for faster configuration and build.
However, the LLVM infrastructures still needs to be installed within the host system.
More in detail, a customized version of LLVM is used to take advantage of the Clang driver library within MARCO.

In order to ease the process for newcomers, the instructions needed to build LLVM are reported here.

The build type is set to `Release` in order to allow for faster build and execution times, but it strongly suggested setting it to `Debug` when developing on MARCO.

The `LLVM_INSTALL_PATH` variable must be set to the desired installation path.

```bash
git clone https://github.com/marco-compiler/llvm-project.git
cd llvm-project
git checkout marco-llvm
mkdir build && cd build

# Set the installation path.
LLVM_INSTALL_PATH=llvm_install_path

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PATH} \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_INSTALL_UTILS=True \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir;openmp" \
  ../llvm

cmake --build . --target install
```

### OpenModelica
In order to relieve MARCO from the object-oriented characteristics of Modelica, the frontend leverages the OpenModelica frontend to obtain a flattened version of the Modelica sources.
The way it is used by MARCO is transparent to the end-user, but yet the compiler must be installed within the system.

The details on how to install OpenModelica are available on its dedicated [website](https://openmodelica.org/).
It is recommended installing the nighly version, because some features leveraged by MARCO may not be present in the stable branch yet.

For macOS users, OpenModelica is available through Homebrew.

### SUNDIALS
The SUNDIALS libraries can usually be installed through package managers.

```bash
sudo apt install libsundials-dev
```

If needed, a build script is also available inside the repository to manually build and install them.
The following dependencies must be installed before attempting a build:

```bash
sudo apt install libgmp-dev libmpc-dev
```

The libraries can then be built by running the `sundials.sh` script inside the `dependencies` folder.

The `SUNDIALS_INSTALL_PATH` variable must be set to the desired installation path.

```bash
# Set the installation path.
SUNDIALS_INSTALL_PATH=sundials_install_path

cd marco/dependencies
./sundials.sh ${SUNDIALS_INSTALL_PATH}
```

### LIT
LLVM's LIT is used to run the regression tests.
It is available through `pip`, the Python's package manager.

```bash
sudo apt install python-pip
pip install lit
```

## Building and installing the runtime libraries
The runtime libraries project provides the libraries to be linked for generating the simulation.

The `RUNTIME_INSTALL_PATH` variable must be set to the desired installation path.

The `LLVM_PATH` variable must be set to the installation path that was used during the LLVM installation process.

By default, the CMake configuration searches for SUNDIALS libraries within the OS.
If the Sundials library have been built manually, the build system of MARCO must be instructed to use them by setting the `MARCO_USE_BUILTIN_SUNDIALS` CMake option must to `ON` and the `SUNDIALS_PATH` variable to their installation path.

The `LLVM_EXTERNAL_LIT` variable represent the path (including the executable name) to the `lit` tool. If it has been installed in user mode, it is usually `/home/user/.local/bin/lit`.

```bash
git clone https://github.com/marco-compiler/marco-runtime.git
cd marco-runtime
mkdir build && cd build

# Set the installation path.
RUNTIME_INSTALL_PATH=runtime_install_path

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${RUNTIME_INSTALL_PATH} \
  -DLLVM_PATH=${LLVM_INSTALL_PATH} \
  ..

cmake --build .

# Run the unit tests.
cmake --build . --target test

# Install the runtime library.
cmake --build . --target install
```

## Building and installing the compiler
With all the requirements set in place, the compiler can be now built through the following procedure.

The `MARCO_INSTALL_PATH` variable must be set to the desired installation path.

The `LIT_PATH` variable must be set to path of the `lit` tool, which is used to run the regression tests.

```bash
cd marco
mkdir build && cd build

# Set the installation path.
MARCO_INSTALL_PATH=marco_install_path

# Set the path of LIT.
LIT_PATH=/home/user/.local/bin/lit

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${MARCO_INSTALL_PATH} \
  -DLLVM_PATH=${LLVM_INSTALL_PATH} \
  -DMARCORuntime_PATH=${RUNTIME_INSTALL_PATH} \
  -DLLVM_EXTERNAL_LIT=${LIT_PATH} \
  ..

cmake --build .

# Run the unit tests.
cmake --build . --target test

# Run the regression tests.
cmake --build . --target check

# Install the compiler.
cmake --build . --target install
```
