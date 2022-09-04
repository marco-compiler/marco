# Dependencies
In this document we list the dependencies of the project and briefly illustrate how to install them.
Their management is left to the user in order to provide complete freedom about the implementation to be used.

The instructions are reported for Ubuntu systems, but can easily be adapted for other operating systems and package managers.

## LLVM
MARCO has been though as a standalone project and thus does not live inside the LLVM repository.
This allows for faster configuration and build.
However, the LLVM infrastructures still needs to be installed within the host system and, due to the absence of MLIR in public packages, there is need for a manual build.
Moreover, as the MLIR project changes rapidly, the referenced LLVM's commit is periodically updated, in order to be always up-to-date with respect to the LLVM project.

In order to ease the process for newcomers, the instructions needed to build LLVM on the required commit are reported here.

The build type is set to `Release` in order to allow for faster build and execution times, but it strongly suggested setting it to `Debug` when developing on MARCO.

The install path is optional, but it strongly recommended setting it to a custom path in order not mix the installed files with system ones or with other installations coming from package managers.

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout ad66bc42b0c377b1bff9841fc4715a17ca947222
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install_path -DLLVM_INSTALL_UTILS=True -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;mlir;openmp" ../llvm
make install
```

### OpenModelica
In order to relieve MARCO from the object-oriented characteristics of Modelica, the frontend leverages the OpenModelica frontend to obtain a flattened version of the Modelica sources. The way it is used by MARCO is transparent to the end-user, but yet the compiler must be installed within the system.

The details on how to install OpenModelica are available on its dedicated [website](https://openmodelica.org/).
It is recommended installing the nighly version, because some features leveraged by MARCO may not be present in the stable branch yet.

For MacOS users, OpenModelica is available through Homebrew.

### Boost

The Boost libraries can be easily retrieved through package managers. For example:

```bash
sudo apt install libboost-all-dev
```

### Sundials
The Sundials libraries can usually be installed through package managers:

```bash
sudo apt install 
```

However, a build script is also available inside the repository in case a manual build is required.
In this case, the following dependencies must be installed before attempting a build:

```bash
sudo apt install libgmp3
sudo apt install libmpc-dev
```

Then it is possible to build the libraries by running the `sundials` script inside the `dependencies` folder:
```bash
cd dependencies
./sundials.sh install_path
```

### LIT
LLVM's LIT is used to run the regression tests.
It is available through pip, the Python's package manager.

```bash
sudo apt install python3-pip
pip3 install lit
```
