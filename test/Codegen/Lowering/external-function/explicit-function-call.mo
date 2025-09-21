// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @explicitCall
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[call:.*]] = bmodelica.external_call @foo(%[[x]]) : (!bmodelica.int) -> !bmodelica.int
// CHECK:       bmodelica.variable_set @y, %[[call]]
// CHECK:   }

function explicitCall
    input Integer x;
    output Integer y;
external "C" y = foo(x);
end explicitCall;
