// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: max: expected between 1 and 2 argument(s) but got 3.

function foo
    input Real x;
    input Real y;
    output Real z;
algorithm
    z := max(x, y, x);
end foo;
