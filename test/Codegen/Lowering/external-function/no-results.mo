// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @noResults
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       bmodelica.external_call @foo(%[[x]]) : (!bmodelica.int) -> ()
// CHECK:   }

function noResults
    output Integer x;
external foo(x);
end noResults;
