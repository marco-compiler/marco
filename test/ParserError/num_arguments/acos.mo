// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: acos: expected 1 argument(s) but got 0.

function foo
    input Real x;
    output Real y;
algorithm
    y := acos();
end foo;
