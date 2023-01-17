# MARCO - Modelica Advanced Research COmpiler
MARCO is a prototype compiler for the Modelica language.
It is written in C++ and leverages both the LLVM and MLIR infrastructures.
It also defines an MLIR dialect that is specific to Modelica.

## Requirements

- LLVM (see the docs for the specific commit to be used)
- OpenModelica
- Sundials libraries
- LLVM's LIT

See the documentation (inside the `docs` folder) for further details and short instructions about how to obtain the required dependencies.

## Building MARCO
Once all the requirements have been fulfilled, MARCO can be built.

The project can be built with the following instructions.
The `LLVM_PATH` variable must be set to the installation path that was used during the LLVM installation process.

If the Sundials library have been built manually, the `SUNDIALS_PATH` variable must be set to their installation path.
In case of libraries installed through package managers, the build system of MARCO must be instructed to use them by setting `MARCO_USE_BUILTIN_SUNDIALS` CMake option to `OFF`.

The `LLVM_EXTERNAL_LIT` variable represent the path (including the executable name) to the `lit` tool. If it has been installed in user mode, it is usually `/home/user/.local/bin/lit`.

```
cd marco
mkdir build && cd build
cmake -DLLVM_PATH=llvm_install_path -DSUNDIALS_PATH=sundials_install_path -DLLVM_EXTERNAL_LIT=lit_executable_path -DCMAKE_BUILD_TYPE=Release ..
make all
```

Doxygen documentation can also be built by adding the `-DMARCO_BUILD_DOCS=ON` option.
Files will be installed inside the `docs` folder.

## Tests
MARCO adopts a test infrastructure that is similar to the one of LLVM, thus defining two categories of tests: unit tests and regression tests.

Unit tests are written using [Google Test](https://github.com/google/googletest/) and Google Mock and are located in the `unittest` directory.
Regression tests, instead, leverage the LIT tool mentioned earlier, and are located in the `test` directory.

In general, unit tests are reserved to ensure the correct behaviour of the data structures, while regression tests are preferred to test broader scopes such as transformation or even the whole compilation pipeline. 

Unit tests can be run with `make test`, while regression tests can be run with `make check`.
