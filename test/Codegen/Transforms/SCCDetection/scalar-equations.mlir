// RUN: modelica-opt %s --split-input-file --detect-scc --canonicalize-model-for-debug | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK:       modelica.main_model {
// CHECK-NEXT:      modelica.scc {
// CHECK-NEXT:          modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:          modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    // x = y
    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // y = 1 - x
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.constant #modelica.int<1>
        %2 = modelica.variable_get @x : !modelica.int
        %3 = modelica.sub %1, %2 : (!modelica.int, !modelica.int) -> !modelica.int
        %4 = modelica.equation_side %0 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}
