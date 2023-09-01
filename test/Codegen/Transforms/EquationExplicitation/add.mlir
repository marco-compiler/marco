// RUN: modelica-opt %s --split-input-file --test-equation-explicitation --canonicalize | FileCheck %s

// First operand.

// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"} {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @y
// CHECK-DAG:       %[[cst:.*]] = modelica.constant #modelica.int<3>
// CHECK-DAG:       %[[sub:.*]] = modelica.sub %[[y]], %[[cst]]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[sub]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<3>
        %2 = modelica.add %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        %3 = modelica.variable_get @y : !modelica.int
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {path = #modelica<equation_path [L, 0, 0]>} : !modelica.equation
}

// -----

// Second operand.

// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"} {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @y
// CHECK-DAG:       %[[cst:.*]] = modelica.constant #modelica.int<3>
// CHECK-DAG:       %[[sub:.*]] = modelica.sub %[[y]], %[[cst]]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[sub]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<3>
        %2 = modelica.add %1, %0 : (!modelica.int, !modelica.int) -> !modelica.int
        %3 = modelica.variable_get @y : !modelica.int
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {path = #modelica<equation_path [L, 0, 1]>} : !modelica.equation
}
