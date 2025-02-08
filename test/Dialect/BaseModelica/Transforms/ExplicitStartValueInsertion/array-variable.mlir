// RUN: modelica-opt %s --split-input-file --insert-missing-start-values | FileCheck %s

// CHECK-LABEL: @Integer

bmodelica.model @Integer {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    // CHECK:       bmodelica.start @x {
    // CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[tensor:.*]] = bmodelica.tensor_broadcast %[[value]]
    // CHECK-NEXT:      bmodelica.yield %[[tensor]]
    // CHECK-NEXT:  }
    // CHECK-SAME:  each = false
    // CHECK-SAME:  fixed = false
    // CHECK-SAME:  implicit = true
}
