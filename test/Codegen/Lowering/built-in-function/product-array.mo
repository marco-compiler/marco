// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.product
// CHECK-SAME: tensor<?x!bmodelica.int> -> !bmodelica.int

function foo
    input Integer[:] x;
    output Integer y;
algorithm
    y := product(x);
end foo;
