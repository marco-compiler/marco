// RUN: modelica-opt %s --index-reduction | FileCheck %s

module {
  bmodelica.model @SimpleArrayIndex0 der = [<@y_1, @der_y_1, {[0,4]}>] {
    bmodelica.variable @q : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @y_1 : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @der_y_1 : !bmodelica.variable<5x!bmodelica.real>

    %0 = bmodelica.equation_template inductions = [%arg0] {
      %3 = bmodelica.constant -1 : index
      %4 = bmodelica.add %arg0, %3 : (index, index) -> index
      %5 = bmodelica.variable_get @q : tensor<5x!bmodelica.real>
      %6 = bmodelica.tensor_extract %5[%4] : tensor<5x!bmodelica.real>
      %7 = bmodelica.time : !bmodelica.real
      %8 = bmodelica.time : !bmodelica.real
      %9 = bmodelica.mul %7, %8 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
      %10 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
      %11 = bmodelica.equation_side %9 : tuple<!bmodelica.real>
      bmodelica.equation_sides %10, %11 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %1 = bmodelica.equation_template inductions = [%arg0] {
      %3 = bmodelica.constant -1 : index
      %4 = bmodelica.add %arg0, %3 : (index, index) -> index
      %5 = bmodelica.variable_get @der_y_1 : tensor<5x!bmodelica.real>
      %6 = bmodelica.tensor_extract %5[%4] : tensor<5x!bmodelica.real>
      %7 = bmodelica.add %arg0, %3 : (index, index) -> index
      %8 = bmodelica.variable_get @q : tensor<5x!bmodelica.real>
      %9 = bmodelica.tensor_extract %8[%7] : tensor<5x!bmodelica.real>
      %10 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
      %11 = bmodelica.equation_side %9 : tuple<!bmodelica.real>
      bmodelica.equation_sides %10, %11 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %2 = bmodelica.equation_template inductions = [%arg0] {
      %3 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
      %4 = bmodelica.constant -1 : index
      %5 = bmodelica.add %arg0, %4 : (index, index) -> index
      %6 = bmodelica.variable_get @y_1 : tensor<5x!bmodelica.real>
      %7 = bmodelica.tensor_extract %6[%5] : tensor<5x!bmodelica.real>
      %8 = bmodelica.equation_side %7 : tuple<!bmodelica.real>
      %9 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
      bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.int>
    }

    bmodelica.initial {
      bmodelica.equation_instance %0, indices = {[1,5]}
      bmodelica.equation_instance %1, indices = {[1,5]}
      bmodelica.equation_instance %2, indices = {[1,5]}
    }

    bmodelica.dynamic {
      bmodelica.equation_instance %0, indices = {[1,5]}
      bmodelica.equation_instance %1, indices = {[1,5]}
    }

    bmodelica.start @der_y_1 {
      %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      %4 = bmodelica.tensor_broadcast %3 : !bmodelica.real -> tensor<5x!bmodelica.real>
      bmodelica.yield %4 : tensor<5x!bmodelica.real>
    } {each = false, fixed = false, implicit = true}

    bmodelica.start @q {
      %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      %4 = bmodelica.tensor_broadcast %3 : !bmodelica.real -> tensor<5x!bmodelica.real>
      bmodelica.yield %4 : tensor<5x!bmodelica.real>
    } {each = false, fixed = false, implicit = true}

    bmodelica.start @y_1 {
      %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      %4 = bmodelica.tensor_broadcast %3 : !bmodelica.real -> tensor<5x!bmodelica.real>
      bmodelica.yield %4 : tensor<5x!bmodelica.real>
    } {each = false, fixed = false, implicit = true}
  }
}

// COM: The model is index 0, so nothing should change.
// CHECK:      bmodelica.model @SimpleArrayIndex0 der = [<@y_1, @der_y_1, {[0,4]}>] {
// CHECK-NEXT:   bmodelica.variable @q : !bmodelica.variable<5x!bmodelica.real>
// CHECK-NEXT:   bmodelica.variable @y_1 : !bmodelica.variable<5x!bmodelica.real>
// CHECK-NEXT:   bmodelica.variable @der_y_1 : !bmodelica.variable<5x!bmodelica.real>

// CHECK-NEXT:   %0 = bmodelica.equation_template inductions = [%arg0] {
// CHECK-NEXT:     %3 = bmodelica.constant -1 : index
// CHECK-NEXT:     %4 = bmodelica.add %arg0, %3 : (index, index) -> index
// CHECK-NEXT:     %5 = bmodelica.variable_get @q : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %6 = bmodelica.tensor_extract %5[%4] : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %7 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %8 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %9 = bmodelica.mul %7, %8 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:     %10 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
// CHECK-NEXT:     %11 = bmodelica.equation_side %9 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %10, %11 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   %1 = bmodelica.equation_template inductions = [%arg0] {
// CHECK-NEXT:     %3 = bmodelica.constant -1 : index
// CHECK-NEXT:     %4 = bmodelica.add %arg0, %3 : (index, index) -> index
// CHECK-NEXT:     %5 = bmodelica.variable_get @der_y_1 : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %6 = bmodelica.tensor_extract %5[%4] : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %7 = bmodelica.add %arg0, %3 : (index, index) -> index
// CHECK-NEXT:     %8 = bmodelica.variable_get @q : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %9 = bmodelica.tensor_extract %8[%7] : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %10 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
// CHECK-NEXT:     %11 = bmodelica.equation_side %9 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %10, %11 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   %2 = bmodelica.equation_template inductions = [%arg0] {
// CHECK-NEXT:     %3 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
// CHECK-NEXT:     %4 = bmodelica.constant -1 : index
// CHECK-NEXT:     %5 = bmodelica.add %arg0, %4 : (index, index) -> index
// CHECK-NEXT:     %6 = bmodelica.variable_get @y_1 : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %7 = bmodelica.tensor_extract %6[%5] : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %8 = bmodelica.equation_side %7 : tuple<!bmodelica.real>
// CHECK-NEXT:     %9 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
// CHECK-NEXT:     bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.int>
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.initial {
// CHECK-NEXT:     bmodelica.equation_instance %0, indices = {[1,5]}
// CHECK-NEXT:     bmodelica.equation_instance %1, indices = {[1,5]}
// CHECK-NEXT:     bmodelica.equation_instance %2, indices = {[1,5]}
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.dynamic {
// CHECK-NEXT:     bmodelica.equation_instance %0, indices = {[1,5]}
// CHECK-NEXT:     bmodelica.equation_instance %1, indices = {[1,5]}
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.start @der_y_1 {
// CHECK-NEXT:     %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     %4 = bmodelica.tensor_broadcast %3 : !bmodelica.real -> tensor<5x!bmodelica.real>
// CHECK-NEXT:     bmodelica.yield %4 : tensor<5x!bmodelica.real>
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}

// CHECK-NEXT:   bmodelica.start @q {
// CHECK-NEXT:     %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     %4 = bmodelica.tensor_broadcast %3 : !bmodelica.real -> tensor<5x!bmodelica.real>
// CHECK-NEXT:     bmodelica.yield %4 : tensor<5x!bmodelica.real>
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}

// CHECK-NEXT:   bmodelica.start @y_1 {
// CHECK-NEXT:     %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     %4 = bmodelica.tensor_broadcast %3 : !bmodelica.real -> tensor<5x!bmodelica.real>
// CHECK-NEXT:     bmodelica.yield %4 : tensor<5x!bmodelica.real>
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}
// CHECK-NEXT: }
