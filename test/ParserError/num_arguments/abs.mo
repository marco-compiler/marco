// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function abs.

function foo
    input Real x;
    output Real y;
algorithm
    y := abs(x, x);
end foo;
