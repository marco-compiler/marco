// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: ndims: expected 1 argument(s) but got 2.

function foo
    input Integer[:,:] x;
    output Integer y;
algorithm
    y := ndims(x, x);
end foo;
