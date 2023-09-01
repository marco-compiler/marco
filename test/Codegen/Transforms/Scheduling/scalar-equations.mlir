// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK:       modelica.scc {
// CHECK-NEXT:      modelica.scheduled_equation_instance %[[t1]] {iteration_directions = [], path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:  }
// CHECK-NEXT:  modelica.scc {
// CHECK-NEXT:      modelica.scheduled_equation_instance %[[t0]] {iteration_directions = [], path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:  }

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
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}
