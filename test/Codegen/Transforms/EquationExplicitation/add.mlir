// RUN: modelica-opt %s --split-input-file --test-equation-explicitation --canonicalize | FileCheck %s

// First operand.

// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK-DAG:       %[[cst:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-DAG:       %[[sub:.*]] = bmodelica.sub %[[y]], %[[cst]]
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[sub]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [L, 0]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.constant #bmodelica.int<3>
        %2 = bmodelica.add %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %3 = bmodelica.variable_get @y : !bmodelica.int
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.main_model {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0, 0]>} : !bmodelica.equation
    }
}

// -----

// Second operand.

// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK-DAG:       %[[cst:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-DAG:       %[[sub:.*]] = bmodelica.sub %[[y]], %[[cst]]
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[sub]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [L, 0]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.constant #bmodelica.int<3>
        %2 = bmodelica.add %1, %0 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %3 = bmodelica.variable_get @y : !bmodelica.int
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.main_model {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0, 1]>} : !bmodelica.equation
    }
}
