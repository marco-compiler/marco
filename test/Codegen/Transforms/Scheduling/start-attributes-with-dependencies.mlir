// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}
// CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}

// CHECK:       bmodelica.schedule @ic {
// CHECK-NEXT:      bmodelica.initial {
// CHECK-NEXT:          bmodelica.scc {
// CHECK-NEXT:              bmodelica.scheduled_equation_instance %2 {iteration_directions = [], path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
// CHECK-NEXT:          }
// CHECK-NEXT:          bmodelica.scc {
// CHECK-NEXT:              bmodelica.scheduled_equation_instance %3 {iteration_directions = [], path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
// CHECK-NEXT:          }
// CHECK-NEXT:          bmodelica.start_equation_instance %0 : !bmodelica.equation
// CHECK-NEXT:          bmodelica.scc {
// CHECK-NEXT:              bmodelica.scheduled_equation_instance %1 {iteration_directions = [], path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @p : !bmodelica.variable<!bmodelica.real, parameter>
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>

    // x = p
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @p : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // x = 1
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y = 20
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.constant #bmodelica.real<20.0> : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // p = y
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @p : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.schedule @ic {
        bmodelica.initial {
            bmodelica.start_equation_instance %t0 : !bmodelica.equation
            bmodelica.scc {
                bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
            }
            bmodelica.scc {
                bmodelica.matched_equation_instance %t2 {path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
            }
            bmodelica.scc {
                bmodelica.matched_equation_instance %t3 {path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
            }
        }
    }
}
