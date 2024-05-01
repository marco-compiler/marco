// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.integer
// CHECK-SAME: !bmodelica.real -> !bmodelica.int

function foo
    input Real x;
    output Real y;
algorithm
    y := integer(x);
end foo;
