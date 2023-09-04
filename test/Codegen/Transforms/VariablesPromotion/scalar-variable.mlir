// RUN: modelica-opt %s --split-input-file --promote-variables-to-parameters | FileCheck %s

// Promotable SCC.

// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: modelica.variable @y : !modelica.variable<!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {initial = true, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {initial = true, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {path = #modelica<equation_path [L, 0]>} : !modelica.equation

    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.variable_get @x : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}
