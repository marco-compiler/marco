// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.rem
// CHECK-SAME: (!bmodelica.int, !bmodelica.int) -> !bmodelica.int

function foo
    input Integer x;
    input Integer y;
    output Integer z;
algorithm
    z := rem(x, y);
end foo;
