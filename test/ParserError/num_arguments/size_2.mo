// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function size at line 10, column 10. Expected between 1 and 2 argument(s) but got 3.

function sizeDimension
    input Real[:,:] x;
    input Integer n;
    output Integer[2] y;
algorithm
    y := size(x, n, x);
end sizeDimension;
