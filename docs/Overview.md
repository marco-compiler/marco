# Overview
The MARCO compiler transforms Modelica / Base Modelica source code into an executable file.

## Translation process
The generation of the executable file is performed in five high-level phases:
- The frontend of the OpenModelica compiler (OMC) is used to instantiate the Modelica model of interest and obtain Base Modelica code.
- The abstract syntax tree (AST) is built and immediately converted to MLIR dialects.
- Language-specific transformations are performed to move from the descriptive nature of the model to the algorithmic one of the simulation. This process can be divided in two sequential high-level steps: the first implements normalization procedures, aiming to simplify the code received by later stages, while the second implements the model solving techniques needed to shift from one paradigm to the other. 
- The mix of MLIR dialects is lowered to the intermediate representation of LLVM (LLVM-IR).
- The already existing infrastructure of LLVM is invoked to generate first the object files, and then link them to appropriate runtime libraries to generate the executable file implementing the simulation. To make this step feasible, modifications to the Clang project have been performed to enable a tight integration of the MARCO frontend with the Clang driver.

## Code organization
From a practical point of view, MARCO is thought as a set of libraries, each of which covering a specific aspect within the compiler.
The organization of the code within the repository mimics the one of LLVM, MLIR and Clang, according to the topic being considered.

Details about each aspect of the compiler can be found in the dedicated documentation files.