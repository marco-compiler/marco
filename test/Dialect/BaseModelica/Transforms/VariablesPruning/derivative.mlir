// RUN: modelica-opt %s --split-input-file --variables-pruning | FileCheck %s

// Output attribute on derived variable.

// CHECK-LABEL: @Test
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

// CHECK: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:  }

bmodelica.model @Test der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, output>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

    // der_x = 1
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @der_x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 1.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>}
    }
}

// -----

// Output attribute on derivative variable.

// CHECK-LABEL: @Test
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, output>

// CHECK: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:  }

bmodelica.model @Test der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, output>

    // der_x = 1
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @der_x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 1.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>}
    }
}
