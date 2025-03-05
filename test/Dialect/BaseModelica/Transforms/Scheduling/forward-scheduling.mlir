// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.dynamic {
// CHECK-NEXT:          bmodelica.scc {
// CHECK-NEXT:              bmodelica.scheduled_equation_instance %[[t1]], indices = {}, match = <@x, {[0,0]}> {iteration_directions = []}
// CHECK-NEXT:          }
// CHECK-NEXT:          bmodelica.scc {
// CHECK-NEXT:              bmodelica.scheduled_equation_instance %[[t0]], indices = {[1,9]}, match = <@x, {[1,9]}> {iteration_directions = [#bmodelica<equation_schedule_direction forward>]}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.int>

    // x[i] = x[i - 1]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %3 = bmodelica.constant 1 : index
        %4 = bmodelica.sub %i0, %3 : (index, index) -> index
        %5 = bmodelica.tensor_extract %2[%4] : tensor<10x!bmodelica.int>
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // x[0] = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.constant 0 : index
        %1 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %2 = bmodelica.tensor_extract %1[%0] : tensor<10x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica<int 0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.schedule @schedule {
        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.matched_equation_instance %t0, indices = {[1,9]}, match = <@x, {[1,9]}>
            }
            bmodelica.scc {
                bmodelica.matched_equation_instance %t1, indices = {}, match = <@x, {[0,0]}>
            }
        }
    }
}