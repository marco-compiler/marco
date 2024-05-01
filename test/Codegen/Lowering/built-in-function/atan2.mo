// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.atan2
// CHECK-SAME: (!bmodelica.real, !bmodelica.real) -> !bmodelica.real

function foo
    input Real y;
    input Real x;
    output Real z;
algorithm
    z := atan2(y, x);
end foo;
