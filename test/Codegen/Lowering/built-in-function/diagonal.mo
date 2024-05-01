// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.diagonal
// CHECK-SAME: !bmodelica.array<?x!bmodelica.int> -> !bmodelica.array<?x?x!bmodelica.int>

function foo
    input Integer[:] x;
    output Integer[:,:] y;
algorithm
    y := diagonal(x);
end foo;
