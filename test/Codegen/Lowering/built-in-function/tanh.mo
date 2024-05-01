// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.tanh
// CHECK-SAME: !bmodelica.real -> !bmodelica.real

function foo
    input Real x;
    output Real y;
algorithm
    y := tanh(x);
end foo;
