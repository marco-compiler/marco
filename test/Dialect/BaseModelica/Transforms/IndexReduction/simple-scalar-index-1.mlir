// RUN: modelica-opt %s --index-reduction | FileCheck %s

module {
  bmodelica.model @SimpleScalarIndex1  {
    bmodelica.variable @q : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y_1 : !bmodelica.variable<!bmodelica.real>

    %0 = bmodelica.equation_template inductions = [] {
      %2 = bmodelica.variable_get @q : !bmodelica.real
      %3 = bmodelica.time : !bmodelica.real
      %4 = bmodelica.time : !bmodelica.real
      %5 = bmodelica.mul %3, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
      %6 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
      %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
      bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %1 = bmodelica.equation_template inductions = [] {
      %2 = bmodelica.variable_get @y_1 : !bmodelica.real
      %3 = bmodelica.variable_get @q : !bmodelica.real
      %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
      %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
      bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.initial {
      bmodelica.equation_instance %0
      bmodelica.equation_instance %1
    }

    bmodelica.dynamic {
      bmodelica.equation_instance %0
      bmodelica.equation_instance %1
    }

    bmodelica.start @q {
      %2 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      bmodelica.yield %2 : !bmodelica.real
    } {each = false, fixed = false, implicit = true}

    bmodelica.start @y_1 {
      %2 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      bmodelica.yield %2 : !bmodelica.real
    } {each = false, fixed = false, implicit = true}
  }
}

// COM: The model is index 1, so nothing should change.
// CHECK:      bmodelica.model @SimpleScalarIndex1  {
// CHECK-NEXT:   bmodelica.variable @q : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:   bmodelica.variable @y_1 : !bmodelica.variable<!bmodelica.real>

// CHECK-NEXT:   %0 = bmodelica.equation_template inductions = [] {
// CHECK-NEXT:     %2 = bmodelica.variable_get @q : !bmodelica.real
// CHECK-NEXT:     %3 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %4 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %5 = bmodelica.mul %3, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:     %6 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
// CHECK-NEXT:     %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   %1 = bmodelica.equation_template inductions = [] {
// CHECK-NEXT:     %2 = bmodelica.variable_get @y_1 : !bmodelica.real
// CHECK-NEXT:     %3 = bmodelica.variable_get @q : !bmodelica.real
// CHECK-NEXT:     %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
// CHECK-NEXT:     %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.initial {
// CHECK-NEXT:     bmodelica.equation_instance %0
// CHECK-NEXT:     bmodelica.equation_instance %1
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.dynamic {
// CHECK-NEXT:     bmodelica.equation_instance %0
// CHECK-NEXT:     bmodelica.equation_instance %1
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.start @q {
// CHECK-NEXT:     %2 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     bmodelica.yield %2 : !bmodelica.real
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}

// CHECK-NEXT:   bmodelica.start @y_1 {
// CHECK-NEXT:     %2 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     bmodelica.yield %2 : !bmodelica.real
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}
// CHECK-NEXT: }
