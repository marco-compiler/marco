// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @implicitCall
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[call:.*]] = bmodelica.external_call @implicitCall(%[[x]]) : (!bmodelica.int) -> !bmodelica.int
// CHECK:       bmodelica.variable_set @y, %[[call]]
// CHECK:   }

function implicitCall
    input Integer x;
    output Integer y;
external "C";
end implicitCall;
