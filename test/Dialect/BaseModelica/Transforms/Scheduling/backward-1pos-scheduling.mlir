// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// COM: for i in 0:8 loop
// COM:     for i in 0:8 loop
// COM:         x[i] = x[i + 1]
// COM:
// COM: for j in 0:9 loop
// COM:     x[9][j] = 0
// COM:
// COM: for i in 0:8 loop
// COM:     x[i][9] = 0

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x10x!bmodelica.int>

    // COM: x[i][j] = x[i + 1][j + 1]
    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x10x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<10x10x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<10x10x!bmodelica.int>
        %3 = bmodelica.constant 1 : index
        %4 = bmodelica.add %i0, %3 : (index, index) -> index
        %5 = bmodelica.add %i1, %3 : (index, index) -> index
        %6 = bmodelica.tensor_extract %2[%4, %5] : tensor<10x10x!bmodelica.int>
        %7 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.int>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [{{.*}}, {{.*}}] attributes {id = "t0"}

    // COM: x[9][j] = 0
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<10x10x!bmodelica.int>
        %1 = bmodelica.constant 9 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.sub %i0, %2 : (index, index) -> index
        %4 = bmodelica.tensor_extract %0[%1, %3] : tensor<10x10x!bmodelica.int>
        %5 = bmodelica.constant #bmodelica<int 0>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [{{.*}}] attributes {id = "t1"}

    // COM: x[i][9] = 0
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @x : tensor<10x10x!bmodelica.int>
        %1 = bmodelica.constant 9 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.sub %i0, %2 : (index, index) -> index
        %4 = bmodelica.tensor_extract %0[%3, %1] : tensor<10x10x!bmodelica.int>
        %5 = bmodelica.constant #bmodelica<int 0>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [{{.*}}] attributes {id = "t2"}

    bmodelica.schedule @schedule {
       bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,8][0,8]>, path = #bmodelica<equation_path [L, 0]>}
            }
            bmodelica.scc {
                bmodelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,9]>, path = #bmodelica<equation_path [L, 0]>}
            }
            bmodelica.scc {
                bmodelica.matched_equation_instance %t2 {indices = #modeling<multidim_range [0,8]>, path = #bmodelica<equation_path [L, 0]>}
            }

            // CHECK-DAG: bmodelica.scheduled_equation_instance %[[t1]]
            // CHECK-DAG: bmodelica.scheduled_equation_instance %[[t2]]

            // CHECK:       bmodelica.scc {
            // CHECK-NEXT:      bmodelica.scheduled_equation_instance %[[t0]]
            // CHECK-SAME:      {
            // CHECK-SAME:          indices = #modeling<multidim_range [0,8][0,8]>
            // CHECK-SAME:          iteration_directions = [#bmodelica<equation_schedule_direction backward_1pos>, #bmodelica<equation_schedule_direction backward_1pos>]
            // CHECK-SAME:      }
            // CHECK-NEXT:  }
        }
    }

    // CHECK-NOT: bmodelica.scheduled_equation_instance
}
