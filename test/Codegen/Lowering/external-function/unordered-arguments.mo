// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @scalar
// CHECK:   bmodelica.external "foo" {
// CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:   %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       bmodelica.yield %[[y]], %[[x]]
// CHECK:   } to {
// CHECK:       %[[z:.*]] = bmodelica.variable_get @z
// CHECK:       bmodelica.yield %[[z]]
// CHECK:   }

function scalar
    input Integer x;
    input Integer y;
    output Integer z;
external z = foo(y, x);
end scalar;
