// RUN: modelica-opt %s --index-reduction | FileCheck %s

module {
  bmodelica.model @SimpleScalarIndex0 der = [<@y_1, @der_y_1>] {
    bmodelica.variable @y_1 : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_y_1 : !bmodelica.variable<!bmodelica.real>

    %0 = bmodelica.equation_template inductions = [] {
      %1 = bmodelica.variable_get @der_y_1 : !bmodelica.real
      %2 = bmodelica.time : !bmodelica.real
      %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
      %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
      bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
      bmodelica.equation_instance %0
    }
  }
}

// COM: The model is index 0, so nothing should change.
// CHECK:      bmodelica.model @SimpleScalarIndex0 der = [<@y_1, @der_y_1>] {
// CHECK-NEXT:   bmodelica.variable @y_1
// CHECK-NEXT:   bmodelica.variable @der_y_1

// CHECK-NEXT:   %0 = bmodelica.equation_template inductions = [] {
// CHECK-NEXT:     %1 = bmodelica.variable_get @der_y_1
// CHECK-NEXT:     %2 = bmodelica.time
// CHECK-NEXT:     %3 = bmodelica.equation_side %1
// CHECK-NEXT:     %4 = bmodelica.equation_side %2
// CHECK-NEXT:     bmodelica.equation_sides %3, %4
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.dynamic {
// CHECK-NEXT:     bmodelica.equation_instance %0
// CHECK-NEXT:   }

// CHECK-NEXT: }
