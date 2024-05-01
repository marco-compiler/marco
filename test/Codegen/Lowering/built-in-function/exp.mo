// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.exp
// CHECK-SAME: !bmodelica.real -> !bmodelica.real

function foo
    input Real x;
    output Real y;
algorithm
    y := exp(x);
end foo;
