// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.linspace
// CHECK-SAME: (!bmodelica.int, !bmodelica.int, !bmodelica.int) -> tensor<?x!bmodelica.real>

function foo
    input Integer start;
    input Integer stop;
    input Integer n;
    output Real[:] y;
algorithm
    y := linspace(start, stop, n);
end foo;
