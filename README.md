# MARCO - Modelica Advanced Research COmpiler
MARCO is a prototype compiler for the Modelica language.
It is written in C++ and leverages both the LLVM and MLIR infrastructures.

## Building MARCO
MARCO runs natively on Linux and macOS. At the moment, Windows is not supported.
Directions to obtain the required dependencies and compile the project can be found [here](docs/BuildOnLinuxMacOS.md).

## Using MARCO
After installation, two executables should appear in the `bin` directory inside the chosen installation path.

### `marco`
The `marco` executable represents the main entry point for the compiler.
The list of options can be obtained by passing the `--help` parameter.

As an example, consider the following model involving a scalar variable and its derivative:

```modelica
model SimpleFirstOrder
    Real x(start = 0, fixed = true);
equation
    der(x) = 1 - x;
end SimpleFirstOrder;
```

The following command takes the source file, named `SimpleFirstOrder.mo`, and generate a binary file named `simulation`.
The numerical integration method is chosen with the `--solver` flag, and in this case consists in the explicit Euler method.
The `<marco_runtime_lib_folder>` argument has to be replaced with the path containing the installed MARCO runtime libraries (tipically, the `lib` folder inside the specified installation path).

```bash
marco SimpleFirstOrder.mo -o simulation --model=SimpleFirstOrder --solver=euler-forward -L <marco_runtime_lib_folder> -Wl,-rpath,<marco_runtime_lib_folder>
```

As the name suggests, the `simulation` binary file implements the simulation that computes the evolution of the system.
Additional parameters can be also be provided when running the simulation. Their list can be obtained using the `--help` argument.

```bash
 ./simulation --time-step=0.5 --end-time=5                                                                                                                                                                  ✔ 
"time","x"
0.000000000,0.000000000
0.500000000,0.500000000
1.000000000,0.750000000
1.500000000,0.875000000
2.000000000,0.937500000
2.500000000,0.968750000
3.000000000,0.984375000
3.500000000,0.992187500
4.000000000,0.996093750
4.500000000,0.998046875
5.000000000,0.999023438
```

### `modelica-opt`
The `modelica-opt` executable is intended to be used mainly for debugging purposes.
It accepts only MLIR code and applies the user-requested pipeline of transformations, thus allowing the user to check their correct behaviour.

## How to contribute
Contributions are welcome and encouraged, both in the form of pull requests and issues reporting bugs, errors, or possible improvements.
