// RUN: modelica-opt %s --split-input-file --variables-pruning | FileCheck %s

// Initial equation with dependency.

// CHECK-LABEL: @InitialWithDependency
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

// CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}
// CHECK: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}

// CHECK:       bmodelica.initial {
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t2]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:  }

// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t3]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:  }

bmodelica.model @InitialWithDependency {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    // x = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y = x
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.variable_get @x : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // z = y
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // z = 0
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @z : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.initial {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.matched_equation_instance %t2 {path = #bmodelica<equation_path [L, 0]>}
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t3 {path = #bmodelica<equation_path [L, 0]>}
    }
}

// -----

// Dyanmic equation with dependency.

// CHECK-LABEL: @DynamicWithDependency
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

// CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}
// CHECK: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}

// CHECK:       bmodelica.initial {
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t2]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:  }

// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.matched_equation_instance %[[t3]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:  }

bmodelica.model @DynamicWithDependency {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    // x = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y = x
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.variable_get @x : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // z = 0
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // z = y
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @z : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.initial {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>}
        bmodelica.matched_equation_instance %t2 {path = #bmodelica<equation_path [L, 0]>}
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t3 {path = #bmodelica<equation_path [L, 0]>}
    }
}
