// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.sin
// CHECK-SAME: !bmodelica.real -> !bmodelica.real

function foo
    input Real x;
    output Real y;
algorithm
    y := sin(x);
end foo;
