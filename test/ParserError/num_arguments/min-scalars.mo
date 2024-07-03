// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function min.

function foo
    input Real x;
    input Real y;
    output Real z;
algorithm
    z := min();
end foo;
