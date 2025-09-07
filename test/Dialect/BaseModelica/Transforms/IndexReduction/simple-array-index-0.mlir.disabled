// RUN: modelica-opt %s --index-reduction | FileCheck %s

module {
  bmodelica.model @SimpleArrayIndex0 der = [<@y_1, @der_y_1, {[0,4]}>] {
    bmodelica.variable @y_1 : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @der_y_1 : !bmodelica.variable<5x!bmodelica.real>

    %0 = bmodelica.equation_template inductions = [%arg0] {
      %1 = bmodelica.constant -1 : index
      %2 = bmodelica.add %arg0, %1 : (index, index) -> index
      %3 = bmodelica.variable_get @der_y_1 : tensor<5x!bmodelica.real>
      %4 = bmodelica.tensor_extract %3[%2] : tensor<5x!bmodelica.real>
      %5 = bmodelica.time : !bmodelica.real
      %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
      %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
      bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
      bmodelica.equation_instance %0, indices = {[1,5]}
    }
  }
}

// COM: The model is index 0, so nothing should change.

// CHECK:      bmodelica.model @SimpleArrayIndex0 der = [<@y_1, @der_y_1, {[0,4]}>] {
// CHECK-NEXT:   bmodelica.variable @y_1 : !bmodelica.variable<5x!bmodelica.real>
// CHECK-NEXT:   bmodelica.variable @der_y_1 : !bmodelica.variable<5x!bmodelica.real>

// CHECK-NEXT:   %0 = bmodelica.equation_template inductions = [%arg0] {
// CHECK-NEXT:     %1 = bmodelica.constant -1 : index
// CHECK-NEXT:     %2 = bmodelica.add %arg0, %1 : (index, index) -> index
// CHECK-NEXT:     %3 = bmodelica.variable_get @der_y_1 : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %4 = bmodelica.tensor_extract %3[%2] : tensor<5x!bmodelica.real>
// CHECK-NEXT:     %5 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
// CHECK-NEXT:     %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.dynamic {
// CHECK-NEXT:     bmodelica.equation_instance %0, indices = {[1,5]}
// CHECK-NEXT:   }
// CHECK-NEXT: }
