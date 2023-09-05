# Building on Linux / macOS
We provide here directions to install ONNX-MLIR on Linux and macOS.
The commands for installing some packages are based on the `apt` tool, but it can be easily adapted to other package managers.

## Requirements
### MLIR
MARCO has been though as a standalone project and thus does not live inside the LLVM repository.
This allows for faster configuration and build.
However, the LLVM infrastructures still needs to be installed within the host system and, due to the absence of MLIR in public packages, there is need for a manual build.
Moreover, as the MLIR project changes rapidly, the referenced LLVM's commit is periodically updated in order to be always up-to-date with respect to MLIR.
In order to ease the process for newcomers, the instructions needed to build LLVM on the required commit are reported here.

The build type is set to `Release` in order to allow for faster build and execution times, but it strongly suggested setting it to `Debug` when developing on MARCO.

The installation path is optional, but it strongly recommended setting it to a custom path in order not mix the installed files with system ones or with other installations coming from package managers.

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 46fab767882d48d2dd46a497baa3197bf9a98ab2
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
However, a build script is available inside the repository to build and install them.
The following dependencies must be installed before attempting a build:

```bash
sudo apt install libgmp3
sudo apt install libmpc-dev
```

The libraries can then be built by running the `sundials.sh` script inside the `dependencies` folder.
Replace the `install_path` keyword with your preferred installation path, which will be used later to build the MARCO runtime library.
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

## Building and installing the runtime library
Despite living in the same repository, the MARCO runtime library is by all means a separate projects that must be built separately and installed before compiling the MARCO compiler.

The `LLVM_PATH` variable must be set to the installation path that was used during the LLVM installation process.

If the Sundials library have been built manually, the `SUNDIALS_PATH` variable must be set to their installation path.
In case of libraries installed through package managers, the build system of MARCO must be instructed to use them by setting `MARCO_USE_BUILTIN_SUNDIALS` CMake option to `OFF`.

The `LLVM_EXTERNAL_LIT` variable represent the path (including the executable name) to the `lit` tool. If it has been installed in user mode, it is usually `/home/user/.local/bin/lit`.

```bash
cd marco/runtime
mkdir build && cd build

# Set the installation path.
RUNTIME_INSTALL_PATH=runtime_install_path

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${RUNTIME_INSTALL_PATH} \
  -DSUNDIALS_PATH=${SUNDIALS_INSTALL_PATH} \
  -DLLVM_PATH=${LLVM_INSTALL_PATH} \
  ..

cmake --build .

# Run the unit tests.
cmake --build . --target test

# Install the runtime library.
cmake --build . --target install
```

## Building and installing the compiler
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
  -DMARCORuntime_DIR=${RUNTIME_INSTALL_PATH}/lib/cmake/MARCORuntime \
  -DLLVM_EXTERNAL_LIT=${LIT_PATH}
  ..

cmake --build .

# Run the unit tests.
cmake --build . --target test

# Run the regression tests.
cmake --build . --target check

# Install the compiler.
cmake --build . --target install
```
