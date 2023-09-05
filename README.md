# MARCO - Modelica Advanced Research COmpiler
MARCO is a prototype compiler for the Modelica language.
It is written in C++ and leverages both the LLVM and MLIR infrastructures.
It also defines an MLIR dialect that is specific to Modelica.

## Building MARCO
MARCO runs natively on Linux and macOS. At the moment, Windows is not supported.
Directions to obtain the required dependencies and compile the project can be found [here](docs/BuildOnLinuxMacOS.md).

## Using MARCO
After installation, two executables should appear in the `bin` directory inside the chosen installation path.

The `marco` executable represents the main entry point for the compiler.
The list of options can be obtained by passing the `--help` parameter.

The `modelica-opt` executable is intended to be used mainly for debugging purposes.
It accepts only MLIR code and applies the user-requested pipeline of transformations, thus allowing the user to check their correct behaviour.
