// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: min: expected between 1 and 2 argument(s) but got 0.

function foo
    input Real x;
    input Real y;
    output Real z;
algorithm
    z := min();
end foo;
