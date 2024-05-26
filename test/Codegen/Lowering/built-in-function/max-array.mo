// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.max
// CHECK-SAME: tensor<?x?x!bmodelica.real> -> !bmodelica.real

function foo
    input Real[:,:] x;
    output Real y;
algorithm
    y := max(x);
end foo;
