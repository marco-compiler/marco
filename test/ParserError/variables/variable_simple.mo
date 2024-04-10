// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown variable identifier inpt_var at line 9, column 19. Did you mean input_var?

function Test2
    input Real input_var;
    output Real output_var;
algorithm
    output_var := inpt_var;
end Test2;
