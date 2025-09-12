// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @scalar
// CHECK:   bmodelica.external "foo" {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       bmodelica.yield %[[x]]
// CHECK:   } to {
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       bmodelica.yield %[[y]]
// CHECK:   }

function scalar
    input Integer[10] x;
    output Integer y;
external y = foo(x);
end scalar;
