// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: identity: expected 1 argument(s) but got 2.

function foo
    input Integer x;
    output Integer[:,:] y;
algorithm
    y := identity(x, x);
end foo;
