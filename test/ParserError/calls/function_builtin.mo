// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown function identifier sze.

function sizeDimension
    input Real[:,:] x;
    input Integer n;
    output Integer[2] y;
algorithm
    y := sze(x, n);
end sizeDimension;
