// RUN: modelica-opt %s --split-input-file --insert-explicit-initial-equations | FileCheck %s

// CHECK:       modelica.model @Test {
// CHECK:           modelica.variable @x : !modelica.variable<!modelica.int>
// CHECK:           %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"} {
// CHECK:               modelica.equation_sides
// CHECK:           }
// CHECK:           modelica.equation_instance %[[t0]] {initial = true}
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %lhs = modelica.equation_side %0 : tuple<!modelica.int>
        %rhs = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %lhs, %rhs : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.equation_instance %t0 {initial = true} : !modelica.equation
}
