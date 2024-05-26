// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.ones
// CHECK-SAME: (!bmodelica.int, !bmodelica.int) -> tensor<?x?x!bmodelica.int>

function foo
    input Integer n1;
    input Integer n2;
    output Real[:] y;
algorithm
    y := ones(n1, n2);
end foo;
