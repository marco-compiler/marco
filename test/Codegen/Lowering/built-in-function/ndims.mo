// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.ndims
// CHECK-SAME: tensor<?x?x!bmodelica.int> -> !bmodelica.int

function foo
    input Integer[:,:] x;
    output Integer y;
algorithm
    y := ndims(x);
end foo;
