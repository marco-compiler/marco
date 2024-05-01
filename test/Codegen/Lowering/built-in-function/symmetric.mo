// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.symmetric
// CHECK-SAME: !bmodelica.array<?x?x!bmodelica.int> -> !bmodelica.array<?x?x!bmodelica.int>

function foo
    input Integer[:,:] x;
    output Integer[:,:] y;
algorithm
    y := symmetric(x);
end foo;