// RUN: modelica-opt %s --split-input-file --equation-explicitation | FileCheck %s

// CHECK-DAG: #[[index_set_0:.*]] = #modeling<index_set {[0,8]}>
// CHECK-DAG: #[[index_set_1:.*]] = #modeling<index_set {[1,9]}>

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.dynamic {
// CHECK-NEXT:          bmodelica.schedule_block {
// CHECK-NEXT:              bmodelica.equation_call @[[eq:.*]] {indices = #modeling<multidim_range [0,8]>}
// CHECK-NEXT:          } {parallelizable = true, readVariables = [#bmodelica.var<@x, #[[index_set_0]]>], writtenVariables = [#bmodelica.var<@x, #[[index_set_1]]>]}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:       bmodelica.equation_function @[[eq]](%[[lb:.*]]: index, %[[ub:.*]]: index) {
// CHECK:           %[[step:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[i0:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:               %[[one:.*]] = arith.constant 1 : index
// CHECK:               %[[i0_remapped:.*]] = arith.addi %[[one]], %[[i0]]
// CHECK:               %[[sub:.*]] = bmodelica.sub %[[i0_remapped]], %{{.*}}
// CHECK:               bmodelica.store %{{.*}}[%[[i0_remapped]]], %{{.*}}
// CHECK:           }
// CHECK-NEXT:      bmodelica.yield
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.int>

    // x[i] = x[i - 1]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<10x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<10x!bmodelica.int>
        %2 = bmodelica.variable_get @x : !bmodelica.array<10x!bmodelica.int>
        %3 = bmodelica.constant 1 : index
        %4 = bmodelica.sub %i0, %3 : (index, index) -> index
        %5 = bmodelica.load %2[%4] : !bmodelica.array<10x!bmodelica.int>
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.schedule @schedule {
        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.scheduled_equation_instance %t0 {indices = #modeling<multidim_range [1,9]>, iteration_directions = [#bmodelica<equation_schedule_direction forward>], path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
            }
        }
    }
}
