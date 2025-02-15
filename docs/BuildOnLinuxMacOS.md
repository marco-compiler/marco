# Building on Linux / macOS
We provide here directions to install MARCO on Linux and macOS.
The commands for installing some packages are based on the `apt` package manager, but they can be easily adapted to others.

## Requirements
### LLVM
MARCO has been though as a standalone project and thus does not live inside the LLVM repository.
This allows for faster configuration and build.
However, the LLVM infrastructures still needs to be installed within the host system.
More in detail, a customized version of LLVM is used to take advantage of the Clang driver library within MARCO.

In order to ease the process for newcomers, the instructions needed to build LLVM are reported here.

The build type is set to `Release` in order to allow for faster build and execution times, but it strongly suggested setting it to `Debug` when developing on MARCO.
The `LLVM_INSTALL_PATH` variable must be set to the desired installation path.
It is also suggested to use Ninja as Makefiles generator, and the `-DLLVM_PARALLEL_{COMPILE,LINK}_JOBS` variables to control the use of machine resources.
See the official LLVM CMake configuration [guide](https://llvm.org/docs/CMake.html) for further details.

```bash
# Use the commit specified in the .jenkins/llvm_version.txt file of the MARCO repository.
LLVM_COMMIT=llvm_commit

# Set the installation path.
LLVM_INSTALL_PATH=llvm_install_path

git clone https://github.com/marco-compiler/llvm-project.git
cd llvm-project
git checkout ${LLVM_COMMIT}
mkdir build && cd build

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PATH} \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_INSTALL_UTILS=True \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir" \
  ../llvm

cmake --build . --target install
```

### OpenModelica
In order to relieve MARCO from the object-oriented features of Modelica, the frontend leverages the OpenModelica frontend to obtain a flattened version of the Modelica sources.

The details on how to install OpenModelica are available on its dedicated [website](https://openmodelica.org/).
It is recommended installing the nightly version, because some features leveraged by MARCO may not be present in the
stable branch yet.

For macOS users, OpenModelica is available through Homebrew.

### LIT
LLVM's LIT is used to run the regression tests.
It is available through `pip`, the Python's package manager.

```bash
sudo apt install python-pip
pip install lit
```

### Runtime libraries

The runtime libraries project provides the libraries to be linked while generating the simulation binary.
They are not strictly needed for building the compiler and, if not installed, the regression test suite of MARCO will
just skip the simulation tests.
The instructions for the compilation and installation of the libraries can be found in
the [dedicated repository](https://github.com/marco-compiler/marco-runtime).

## Building and installing the compiler
With all the requirements set in place, the compiler can be now built through the following procedure.

To use the script shown below you have to set or replace the following environment variables:

| Name                 | Description                                                           |
|:---------------------|:----------------------------------------------------------------------|
| `MARCO_INSTALL_PATH` | Path to final installation directory of the compiler                  |
| `MARCO_RUNTIME_PATH` | Path to the installation directory of the runtime libraries           |
| `LLVM_INSTALL_PATH`  | Path to the installation of the adapted LLVM project (see LLVM above) |

Furthermore, the following CMake variables can optionally be used:

| Name                 | Description                                                 |
|:---------------------|:------------------------------------------------------------|
| `Python3_EXECUTABLE` | Path to the Python interpreter                              |
| `LLVM_EXTERNAL_LIT`  | Path to the executable LLVM Integrated Tester (`lit`)       |
| `MARCO_SANITIZER`    | The sanitizer to be used for the compiler (e.g., `address`) |

```bash
cd marco
mkdir build && cd build

# Set the installation path.
MARCO_INSTALL_PATH=marco_install_path

# Set the path of the runtime libraries.
# Remove the following line and the correspondent variable in the CMake invocation if the libraries have not been installed.
MARCO_RUNTIME_PATH=marco_runtime_install_path

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${MARCO_INSTALL_PATH} \
  -DLLVM_PATH=${LLVM_INSTALL_PATH} \
  -DMARCO_RUNTIME_PATH=${RUNTIME_INSTALL_PATH} \
  ..

cmake --build .

# Run the unit tests.
cmake --build . --target test

# Run the regression tests.
cmake --build . --target check

# Install the compiler.
cmake --build . --target install
```
