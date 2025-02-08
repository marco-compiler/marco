// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    // COM: x = y
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.variable_get @y : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: y = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.schedule @schedule {
        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>}
            }
            bmodelica.scc {
                bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>}
            }

            // CHECK:       bmodelica.scc {
            // CHECK-NEXT:      bmodelica.scheduled_equation_instance %[[t1]]
            // CHECK-SAME:      {
            // CHECK-SAME:          iteration_directions = []
            // CHECK-SAME:      }
            // CHECK-NEXT:  }
            // CHECK:       bmodelica.scc {
            // CHECK-NEXT:      bmodelica.scheduled_equation_instance %[[t0]]
            // CHECK-SAME:      {
            // CHECK-SAME:          iteration_directions = []
            // CHECK-SAME:      }
            // CHECK-NEXT:  }
        }
    }

    // CHECK-NOT: bmodelica.scheduled_equation_instance
}
