// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [] attributes {id = "t2"}
// CHECK-DAG: %[[t3:.*]] = modelica.equation_template inductions = [] attributes {id = "t3"}

// CHECK:       modelica.schedule @ic {
// CHECK-NEXT:      modelica.initial_model {
// CHECK-NEXT:          modelica.scc {
// CHECK-NEXT:              modelica.scheduled_equation_instance %2 {iteration_directions = [], path = #modelica<equation_path [L, 0]>} : !modelica.equation
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.scc {
// CHECK-NEXT:              modelica.scheduled_equation_instance %3 {iteration_directions = [], path = #modelica<equation_path [L, 0]>} : !modelica.equation
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.start_equation_instance %0 : !modelica.equation
// CHECK-NEXT:          modelica.scc {
// CHECK-NEXT:              modelica.scheduled_equation_instance %1 {iteration_directions = [], path = #modelica<equation_path [L, 0]>} : !modelica.equation
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @p : !modelica.variable<!modelica.real, parameter>
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>

    // x = p
    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @p : !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // x = 1
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.constant #modelica.real<1.0> : !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // y = 20
    %t2 = modelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = modelica.variable_get @y : !modelica.real
        %1 = modelica.constant #modelica.real<20.0> : !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // p = y
    %t3 = modelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = modelica.variable_get @p : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.schedule @ic {
        modelica.initial_model {
            modelica.start_equation_instance %t0 : !modelica.equation
            modelica.scc {
                modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
            }
            modelica.scc {
                modelica.matched_equation_instance %t2 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
            }
            modelica.scc {
                modelica.matched_equation_instance %t3 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
            }
        }
    }
}
