// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: asin: expected 1 argument(s) but got 2.

function foo
    input Real x;
    output Real y;
algorithm
    y := asin(x, x);
end foo;
