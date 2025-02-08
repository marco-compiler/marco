// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// COM: Binding equation for a scalar parameter.

// CHECK-LABEL: @scalarParameter

bmodelica.model @scalarParameter {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, parameter>

    bmodelica.binding_equation @x {
      %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
      bmodelica.yield %0 : !bmodelica.int
    }

    // CHECK:       bmodelica.start @x {
    // CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      bmodelica.yield %[[value]]
    // CHECK-NEXT:  }
    // CHECK-SAME:  each = false
    // CHECK-SAME:  fixed = true
}

// -----

// COM: Binding equation for an array parameter.

// CHECK-LABEL: @arrayParameter

bmodelica.model @arrayParameter {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>

    bmodelica.binding_equation @x {
      %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
      %1 = bmodelica.tensor_broadcast %0: !bmodelica.int -> tensor<3x!bmodelica.int>
      bmodelica.yield %1 : tensor<3x!bmodelica.int>
    }

    // CHECK:       bmodelica.start @x {
    // CHECK-NEXT:      %[[el:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-NEXT:      %[[value:.*]] = bmodelica.tensor_broadcast %[[el]]
    // CHECK-NEXT:      bmodelica.yield %[[value]]
    // CHECK-NEXT:  }
    // CHECK-SAME:  each = false
    // CHECK-SAME:  fixed = true
}
