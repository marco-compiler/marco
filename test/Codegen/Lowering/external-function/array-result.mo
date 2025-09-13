// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @scalar
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[call:.*]] = bmodelica.external_call @foo(%[[x]]) : (!bmodelica.int) -> tensor<10x!bmodelica.int>
// CHECK:       bmodelica.variable_set @y, %[[call]]
// CHECK:   }

function scalar
    input Integer x;
    output Integer[10] y;
external y = foo(x);
end scalar;
