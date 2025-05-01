// RUN: modelica-opt %s --index-reduction | FileCheck %s

module {
  bmodelica.model @SimpleScalarIndex1  {
    bmodelica.variable @y_1 : !bmodelica.variable<!bmodelica.real>

    %0 = bmodelica.equation_template inductions = [] {
      %1 = bmodelica.variable_get @y_1 : !bmodelica.real
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

// COM: The model is index 1, so nothing should change.
// CHECK:      bmodelica.model @SimpleScalarIndex1  {
// CHECK-NEXT:   bmodelica.variable @y_1 : !bmodelica.variable<!bmodelica.real>

// CHECK-NEXT:   %0 = bmodelica.equation_template inductions = [] {
// CHECK-NEXT:     %1 = bmodelica.variable_get @y_1 : !bmodelica.real
// CHECK-NEXT:     %2 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
// CHECK-NEXT:     %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.dynamic {
// CHECK-NEXT:     bmodelica.equation_instance %0
// CHECK-NEXT:   }
// CHECK-NEXT: }
