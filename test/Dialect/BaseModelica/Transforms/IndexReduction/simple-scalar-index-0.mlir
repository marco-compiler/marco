// RUN: modelica-opt %s --index-reduction | FileCheck %s

module {
  bmodelica.model @SimpleScalarIndex0 der = [<@y_1, @der_y_1>] {
    bmodelica.variable @q : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y_1 : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_y_1 : !bmodelica.variable<!bmodelica.real>

    %0 = bmodelica.equation_template inductions = [] {
      %3 = bmodelica.variable_get @q : !bmodelica.real
      %4 = bmodelica.time : !bmodelica.real
      %5 = bmodelica.time : !bmodelica.real
      %6 = bmodelica.mul %4, %5 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
      %7 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
      %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
      bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %1 = bmodelica.equation_template inductions = [] {
      %3 = bmodelica.variable_get @der_y_1 : !bmodelica.real
      %4 = bmodelica.variable_get @q : !bmodelica.real
      %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
      %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
      bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %2 = bmodelica.equation_template inductions = [] {
      %3 = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
      %4 = bmodelica.variable_get @y_1 : !bmodelica.real
      %5 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
      %6 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
      bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.int>
    }

    bmodelica.initial {
      bmodelica.equation_instance %0
      bmodelica.equation_instance %1
      bmodelica.equation_instance %2
    }

    bmodelica.dynamic {
      bmodelica.equation_instance %1
      bmodelica.equation_instance %2
    }

    bmodelica.start @der_y_1 {
      %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      bmodelica.yield %3 : !bmodelica.real
    } {each = false, fixed = false, implicit = true}

    bmodelica.start @q {
      %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      bmodelica.yield %3 : !bmodelica.real
    } {each = false, fixed = false, implicit = true}

    bmodelica.start @y_1 {
      %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      bmodelica.yield %3 : !bmodelica.real
    } {each = false, fixed = false, implicit = true}
  }
}

// COM: The model is index 0, so nothing should change.
// CHECK:      bmodelica.model @SimpleScalarIndex0 der = [<@y_1, @der_y_1>] {
// CHECK-NEXT:   bmodelica.variable @q : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:   bmodelica.variable @y_1 : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:   bmodelica.variable @der_y_1 : !bmodelica.variable<!bmodelica.real>

// CHECK-NEXT:   %0 = bmodelica.equation_template inductions = [] {
// CHECK-NEXT:     %3 = bmodelica.variable_get @q : !bmodelica.real
// CHECK-NEXT:     %4 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %5 = bmodelica.time : !bmodelica.real
// CHECK-NEXT:     %6 = bmodelica.mul %4, %5 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:     %7 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
// CHECK-NEXT:     %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   %1 = bmodelica.equation_template inductions = [] {
// CHECK-NEXT:     %3 = bmodelica.variable_get @der_y_1 : !bmodelica.real
// CHECK-NEXT:     %4 = bmodelica.variable_get @q : !bmodelica.real
// CHECK-NEXT:     %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
// CHECK-NEXT:     %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
// CHECK-NEXT:     bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
// CHECK-NEXT:   }

// CHECK-NEXT:   %2 = bmodelica.equation_template inductions = [] {
// CHECK-NEXT:     %3 = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
// CHECK-NEXT:     %4 = bmodelica.variable_get @y_1 : !bmodelica.real
// CHECK-NEXT:     %5 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
// CHECK-NEXT:     %6 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
// CHECK-NEXT:     bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.int>
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.initial {
// CHECK-NEXT:     bmodelica.equation_instance %0
// CHECK-NEXT:     bmodelica.equation_instance %1
// CHECK-NEXT:     bmodelica.equation_instance %2
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.dynamic {
// CHECK-NEXT:     bmodelica.equation_instance %1
// CHECK-NEXT:     bmodelica.equation_instance %2
// CHECK-NEXT:   }

// CHECK-NEXT:   bmodelica.start @der_y_1 {
// CHECK-NEXT:     %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     bmodelica.yield %3 : !bmodelica.real
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}

// CHECK-NEXT:   bmodelica.start @q {
// CHECK-NEXT:     %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     bmodelica.yield %3 : !bmodelica.real
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}

// CHECK-NEXT:   bmodelica.start @y_1 {
// CHECK-NEXT:     %3 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:     bmodelica.yield %3 : !bmodelica.real
// CHECK-NEXT:   } {each = false, fixed = false, implicit = true}
// CHECK-NEXT: }
