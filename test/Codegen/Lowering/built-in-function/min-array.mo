// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.min
// CHECK-SAME: tensor<?x?x!bmodelica.real> -> !bmodelica.real

function foo
    input Real[:,:] x;
    output Real y;
algorithm
    y := min(x);
end foo;
