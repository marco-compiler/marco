// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.max
// CHECK-SAME: (!bmodelica.real, !bmodelica.real) -> !bmodelica.real

function foo
    input Real x;
    input Real y;
    output Real z;
algorithm
    z := max(x, y);
end foo;
