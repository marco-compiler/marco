// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Unknown function identifier sze. Did you mean size?

function sizeDimension
    input Real[:,:] x;
    input Integer n;
    output Integer[2] y;
algorithm
    y := sze(x, n);
end sizeDimension;
