# Dialects
Most of the MARCO compilation pipeline is based on MLIR.
Dialects have been defined to represent the information needed by the compiler throughout the compilation pipeline.
Some dialects. Depending on the dialect, these multiple levels of abstractions may also appear within the operations of the dialect itself, with some being designed to capture the original language semantics while others being more code-generation oriented.

## Base Modelica dialect
As the name suggests, the `bmodelica` dialect represents the MLIR counterpart of the language.
Types and attributes are implemented to properly represent Base Modelica compile-time knowledge.
For example, the `!bmodelica.real` type maps to the `Real` Base Modelica type.

Also many operations are designed to retain as much as possible the high-level constructs and semantics.
An example is the `bmodelica.function` operation, which may contain variable and algorithm declarations:

```mlir
bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>
    
    bmodelica.algorithm {
       %0 = bmodelica.varialbe_get @x : !bmodelica.real
       bmodelica.variable_set @y, %0 : !bmodelica.real
    }
}
```

Other operations, however, are designed to ease the lowering process and better integrate with the existing MLIR optimizations, sometimes sacrificing some high-level semantics.
For example, the `bmodelica.raw_function` operation is obtained starting from `bmodelica.raw_function`, but the variable declarations are converted to function arguments / return values.

```mlir
bmodelica.raw_function @foo(%arg0: !modelica.real>) -> !bmodelica.real {
    bmodelica.raw_return %arg0 : !bmodelica.real
}
```

# Modeling dialect
The `modeling` dialect contains the attributes used to efficiently represent set of indices.
It is the MLIR counterpart of the Modeling library of MARCO, which is designed independently from the MLIR dialects.

## Runtime dialect
The `runtime` dialect represents the way to interface with the MARCO runtime library.

## SUNDIALS dialect
The `sundials` dialect contains the operation, types and attributes that are common to all the dialects aiming to interface with the SUNDIALS libraries.
Those dialects, namely `ida` and `kinsol`, are allowed -- and supposed -- to use them to reduce code duplication.

## IDA dialect
The `ida` dialect provides a decoupling mechanism to separate the APIs exposed by the IDA solver (or, better, the APIs of the MARCO runtime libraries interfacing to them) and the modeling concepts it entails.

For example, `ida.residual_function` is a function-like operation that computes the residual value of an equation.
The body of its unique region implements the computation logic, but the low-level details about library interfacing are not exposed.

```mlir
// The arguments represent the scalar indices of the array equation.
// No indices are given in case of scalar equations.
ida.residual_function @res_0(%arg0: index, %arg1: index) {
    %residual_value = ... : f64
    ida.return %residual_value : f64
}
```

## KINSOL dialect
The `kinsol` dialect serves the same scope of `ida`, but with respect to the KINSOL solver.
