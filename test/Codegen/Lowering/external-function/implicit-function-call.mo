// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @scalar
// CHECK:   bmodelica.external "scalar" {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       bmodelica.yield %[[x]]
// CHECK:   } to {
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       bmodelica.yield %[[y]]
// CHECK:   }

function scalar
    input Integer x;
    output Integer y;
external;
end scalar;
