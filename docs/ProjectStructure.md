# Project structure
The project is thought as a set of libraries, each of which covering a specific topic.
Most of them rely on the LLVM infrastructure and due to this dependency the directories organization follows the patterns of LLVM itself.

## AST library
It contains the data structures used to keep an in-memory representation of the Modelica source code.
At the moment the library also contains the parser and the validation / modification passes, even though this may be in future revised in order to make the library contain only the AST representation.

## Codegen library
It contains the MLIR dialects that are used within MARCO, together with the transformations needed to convert them to LLVM-IR.
As explained later, this library will be refactored in order to contain just the plain code generation, while the dialects and the transformation passes will be moved to dedicated libraries. 

## Diagnostic library
The library contains utility classes and methods to provide a uniform way of printing diagnostic messages. 

## Dialects library
The library has been just currently created with the effort of extracting the dialects definition from the codegen library and to make them reflecting the best practices of MLIR.
For this reason no other library is currently using it and can be considered as a work-in-progress.

## Modeling library
It contains the algorithms used to operate on the models being processed.
The library has been designed to be independent from MLIR dialects, code generation, or any data structure.

## Frontend library
It contains the frontend the user will interface with. It handles the user inputs and setup the whole compilation pipeline.

## Runtime library
The runtime library represent a special case inside MARCO, as it is not used by MARCO itself but rather linked to the generated simulation.
In fact, the library provides useful methods to be used at simulation runtime.

## Utils library
It contains utility classes used within the whole compiler.
