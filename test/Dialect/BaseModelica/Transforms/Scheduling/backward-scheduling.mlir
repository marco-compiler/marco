// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// for i in 0:8 loop
//    x[i] = x[i + 1]
//
// x[9] = 0

// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.dynamic {
// CHECK-NEXT:          bmodelica.scc {
// CHECK-NEXT:              bmodelica.scheduled_equation_instance %[[t1]], match = <@x, {[9,9]}> {iteration_directions = []}
// CHECK-NEXT:          }
// CHECK-NEXT:          bmodelica.scc {
// CHECK-NEXT:              bmodelica.scheduled_equation_instance %[[t0]], match = <@x, {[0,8]}> {indices = #modeling<multidim_range [0,8]>, iteration_directions = [#bmodelica<equation_schedule_direction backward>]}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.int>

    // x[i] = x[i + 1]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %3 = bmodelica.constant 1 : index
        %4 = bmodelica.add %i0, %3 : (index, index) -> index
        %5 = bmodelica.tensor_extract %2[%4] : tensor<10x!bmodelica.int>
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // x[9] = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %1 = bmodelica.constant 9 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<10x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica<int 0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.schedule @schedule {
       bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.matched_equation_instance %t0, indices = {[0,8]}, match = <@x, {[0,8]}>
            }
            bmodelica.scc {
                bmodelica.matched_equation_instance %t1, indices = {}, match = <@x, {[9,9]}>
            }
        }
    }
}
