// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.sum
// CHECK-SAME: !bmodelica.array<?x!bmodelica.int> -> !bmodelica.int

function foo
    input Integer[:] x;
    output Integer y;
algorithm
    y := sum(x);
end foo;
