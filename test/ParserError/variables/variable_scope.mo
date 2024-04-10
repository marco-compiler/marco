// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown variable identifier input_var at line 16, column 20. Did you mean input_var2?

function Test1
    input Real input_var;
    output Real output_var;
algorithm
    output_var := input_var;
end Test1;

function Test2
    input Real input_var2;
    output Real output_var2;
algorithm
    output_var2 := input_var;
end Test2;
