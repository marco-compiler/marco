# MARCO - Modelica Advanced Research COmpiler
MARCO is a prototype compiler for the [Modelica](https://modelica.org/) language, and is based on the [LLVM / MLIR](https://mlir.llvm.org/) compiler technology.

## Using MARCO
Packages are not yet available.
The easiest to use MARCO is through the [prebuilt Docker image](https://github.com/marco-compiler/marco/pkgs/container/marco-prod-debian-12).
The image provides all the dependencies and the compiler itself.

Two Python scripts, located in the `docker` folder, are provided to conveniently run the compiler and the generated simulation inside a Docker container.
 - `marco.py` invokes the `marco` tool, which represents the entry point of the compiler. The list of options can be obtained by passing the `--help` parameter.
 - `run-sim.py` runs a simulation binary.

As an example, consider the following model involving a scalar variable and its derivative:

```modelica
model SimpleFirstOrder
    Real x(start = 0, fixed = true);
equation
    der(x) = 1 - x;
end SimpleFirstOrder;
```

The following command takes the source file, named `SimpleFirstOrder.mo`, and generates a binary file named `simulation`.
The numerical integration method is chosen with the `--solver` flag, and in this case consists in the forward Euler method.

```bash
./marco.py SimpleFirstOrder.mo -o simulation --model=SimpleFirstOrder --solver=euler-forward
```

As the name suggests, the `simulation` binary file implements the simulation that computes the evolution of the system.
Additional parameters can be also be provided when running the simulation. Their list can be obtained using the `--help` argument.

```bash
./run-sim.py ./simulation --time-step=0.5 --end-time=5                                                                                                                                                                  ✔ 

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

## Building MARCO
Two development Docker images, `marco-dev-debug-debian-12` and `marco-dev-release-debian-12`, are made available.
Both the images provide all the dependencies that are needed to compile MARCO, but do not contain the compiler itself.
The images different in the build characteristics of the installed software:

|                               | LLVM                 | Runtime libraries | OpenModelica  |
|:------------------------------|:---------------------|:------------------|:--------------|
| `marco-dev-debug-debian-12`   | Release + assertions | Debug             | Release       |
| `marco-dev-release-debian-12` | Release              | Release           | Release       |

A Visual Studio Code Dev Container can be set up using the [`docs/devcontainer.json`](docs/devcontainer.json) file.

Of course, it is also possible to build MARCO natively.
Directions to obtain the required dependencies and compile the project can be found [here](docs/BuildOnLinuxMacOS.md).

## Contributing
This project is licensed under the GNU Lesser General Public License (LGPL).
Under the terms of this license, you are free to use, study, modify, and redistribute the code, provided you comply with the obligations of the LGPL.

By contributing, you acknowledge and agree that your contributions are provided under the same LGPL license as the rest of the project, unless explicitly stated otherwise.

### Contribution Policy

1. **Upstream repository control**  
   The copyright holders and maintainers retain full control over the contents of the official repository.  
   Submitting a patch, pull request, or other contribution does not imply acceptance.

2. **Modifications outside this repository**  
   You are free to make and distribute modifications in your own copies or forks, provided that such distribution complies with the LGPL (including, but not limited to, providing source code and preserving notices).

3. **Acceptance criteria**  
   Contributions will be reviewed on technical, legal, and project-governance grounds.  
   Contributions may be accepted, modified, or rejected at the maintainers' discretion.

### Reporting Issues and Suggesting Enhancements

We welcome and encourage community participation.  
You can contribute not only through pull requests but also by reporting bugs, documenting errors, or suggesting improvements.

If you have ideas for new features, please share them in the dedicated [discussion category](https://github.com/marco-compiler/marco/discussions/categories/features).
