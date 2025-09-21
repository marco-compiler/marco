// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @arrayResult
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[call:.*]] = bmodelica.external_call @foo(%[[x]]) : (!bmodelica.int) -> tensor<10x!bmodelica.int>
// CHECK:       bmodelica.variable_set @y, %[[call]]
// CHECK:   }

function arrayResult
    input Integer x;
    output Integer[10] y;
external "C" y = foo(x);
end arrayResult;
