// RUN: modelica-opt %s --index-reduction | FileCheck %s

module {
  bmodelica.model @SimpleArrayIndex1  {
    bmodelica.variable @q : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @y_1 : !bmodelica.variable<5x!bmodelica.real>

    %0 = bmodelica.equation_template inductions = [%arg0] {
      %2 = bmodelica.constant -1 : index
      %3 = bmodelica.add %arg0, %2 : (index, index) -> index
      %4 = bmodelica.variable_get @q : tensor<5x!bmodelica.real>
      %5 = bmodelica.tensor_extract %4[%3] : tensor<5x!bmodelica.real>
      %6 = bmodelica.time : !bmodelica.real
      %7 = bmodelica.time : !bmodelica.real
      %8 = bmodelica.mul %6, %7 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
      %9 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
      %10 = bmodelica.equation_side %8 : tuple<!bmodelica.real>
      bmodelica.equation_sides %9, %10 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %1 = bmodelica.equation_template inductions = [%arg0] {
      %2 = bmodelica.constant -1 : index
      %3 = bmodelica.add %arg0, %2 : (index, index) -> index
      %4 = bmodelica.variable_get @y_1 : tensor<5x!bmodelica.real>
      %5 = bmodelica.tensor_extract %4[%3] : tensor<5x!bmodelica.real>
      %6 = bmodelica.add %arg0, %2 : (index, index) -> index
      %7 = bmodelica.variable_get @q : tensor<5x!bmodelica.real>
      %8 = bmodelica.tensor_extract %7[%6] : tensor<5x!bmodelica.real>
      %9 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
      %10 = bmodelica.equation_side %8 : tuple<!bmodelica.real>
      bmodelica.equation_sides %9, %10 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.initial {
      bmodelica.equation_instance %0, indices = {[1,5]}
      bmodelica.equation_instance %1, indices = {[1,5]}
    }

    bmodelica.dynamic {
      bmodelica.equation_instance %0, indices = {[1,5]}
      bmodelica.equation_instance %1, indices = {[1,5]}
    }

    bmodelica.start @q {
      %2 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      %3 = bmodelica.tensor_broadcast %2 : !bmodelica.real -> tensor<5x!bmodelica.real>
      bmodelica.yield %3 : tensor<5x!bmodelica.real>
    } {each = false, fixed = false, implicit = true}

    bmodelica.start @y_1 {
      %2 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      %3 = bmodelica.tensor_broadcast %2 : !bmodelica.real -> tensor<5x!bmodelica.real>
      bmodelica.yield %3 : tensor<5x!bmodelica.real>
    } {each = false, fixed = false, implicit = true}
  }
}

// COM: The model is index 1, so nothing should change.

// CHECK:      bmodelica.model @SimpleArrayIndex1  {
// CHECK-NEXT:   bmodelica.variable @q : !bmodelica.variable<5x!bmodelica.real>
// CHECK-NEXT:   bmodelica.variable @y_1 : !bmodelica.variable<5x!bmodelica.real>

// CHECK-NEXT:   %0 = bmodelica.equation_template inductions = [%arg0] {
// CHECK-NEXT:     %2 = bmodelica.constant -1 : index
// CHECK-NEXT:     %3 = bmodelica.add %arg0, %2 : (index, index) -> index
// CHECK-NEXT:     %4 = bmodelica.variable_get @q : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %5 = bmodelica.tensor_extract %4[%3] : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %6 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %7 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %8 = bmodelica.mul %6, %7 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:     %9 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
// CHECK-NEXT:     %10 = bmodelica.equation_side %8 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %9, %10 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   %1 = bmodelica.equation_template inductions = [%arg0] {
// CHECK-NEXT:     %2 = bmodelica.constant -1 : index
// CHECK-NEXT:     %3 = bmodelica.add %arg0, %2 : (index, index) -> index
// CHECK-NEXT:     %4 = bmodelica.variable_get @y_1 : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %5 = bmodelica.tensor_extract %4[%3] : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %6 = bmodelica.add %arg0, %2 : (index, index) -> index
// CHECK-NEXT:     %7 = bmodelica.variable_get @q : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %8 = bmodelica.tensor_extract %7[%6] : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %9 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
// CHECK-NEXT:     %10 = bmodelica.equation_side %8 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %9, %10 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.initial {
// CHECK-NEXT:     bmodelica.equation_instance %0, indices = {[1,5]}
// CHECK-NEXT:     bmodelica.equation_instance %1, indices = {[1,5]}
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.dynamic {
// CHECK-NEXT:     bmodelica.equation_instance %0, indices = {[1,5]}
// CHECK-NEXT:     bmodelica.equation_instance %1, indices = {[1,5]}
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.start @q {
// CHECK-NEXT:     %2 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     %3 = bmodelica.tensor_broadcast %2 : !bmodelica.real -> tensor<5x!bmodelica.real>
// CHECK-NEXT:     bmodelica.yield %3 : tensor<5x!bmodelica.real>
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}

// CHECK-NEXT:   bmodelica.start @y_1 {
// CHECK-NEXT:     %2 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     %3 = bmodelica.tensor_broadcast %2 : !bmodelica.real -> tensor<5x!bmodelica.real>
// CHECK-NEXT:     bmodelica.yield %3 : tensor<5x!bmodelica.real>
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}
// CHECK-NEXT: }
