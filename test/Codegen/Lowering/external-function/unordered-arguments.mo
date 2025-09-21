// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @unorderedArguments
// CHECK:   bmodelica.algorithm {
// CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:   %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       %[[call:.*]] = bmodelica.external_call @foo(%[[y]], %[[x]]) : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
// CHECK:       bmodelica.variable_set @z, %[[call]]
// CHECK:   }

function unorderedArguments
    input Integer x;
    input Integer y;
    output Integer z;
external "C" z = foo(y, x);
end unorderedArguments;
