// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function zeros.

function foo
    input Integer n1;
    input Integer n2;
    output Real[:] y;
algorithm
    y := zeros();
end foo;
