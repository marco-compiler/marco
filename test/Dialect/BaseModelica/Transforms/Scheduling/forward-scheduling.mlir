// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// COM: x[0] = 0
// COM: x[1] = 0
// COM:
// COM: for i in 2:9 loop
// COM:    x[i] = x[i - 2]

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.int>

    // COM: x[i] = x[i - 2]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.sub %i0, %3 : (index, index) -> index
        %5 = bmodelica.tensor_extract %2[%4] : tensor<10x!bmodelica.int>
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [{{.*}}] attributes {id = "t0"}

    // COM: x[0] = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.constant 0 : index
        %1 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %2 = bmodelica.tensor_extract %1[%0] : tensor<10x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica<int 0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: x[1] = 0
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.constant 1 : index
        %1 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %2 = bmodelica.tensor_extract %1[%0] : tensor<10x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica<int 0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    bmodelica.schedule @schedule {
        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [2,9]>, path = #bmodelica<equation_path [L, 0]>}
            }
            bmodelica.scc {
                bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>}
            }
            bmodelica.scc {
                bmodelica.matched_equation_instance %t2 {path = #bmodelica<equation_path [L, 0]>}
            }

            // CHECK-DAG:   bmodelica.scheduled_equation_instance %[[t1]] {iteration_directions = [], path = #bmodelica<equation_path [L, 0]>}
            // CHECK-DAG:   bmodelica.scheduled_equation_instance %[[t2]] {iteration_directions = [], path = #bmodelica<equation_path [L, 0]>}
            // CHECK:       bmodelica.scc {
            // CHECK-NEXT:      bmodelica.scheduled_equation_instance %[[t0]]
            // CHECK-SAME:      {
            // CHECK-SAME:          indices = #modeling<multidim_range [2,9]>
            // CHECK-SAME:          iteration_directions = [#bmodelica<equation_schedule_direction forward>]
            // CHECK-SAME:      }
            // CHECK-NEXT:  }
        }
    }

    // CHECK-NOT: bmodelica.scheduled_equation_instance
}
