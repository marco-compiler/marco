// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function floor.

function foo
    input Real x;
    output Real y;
algorithm
    y := floor(x, x);
end foo;
