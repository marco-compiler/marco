// RUN: modelica-opt %s --split-input-file --equation-explicitation | FileCheck %s

// CHECK-DAG: #[[index_set_0:.*]] = #modeling<index_set {[0,8]}>
// CHECK-DAG: #[[index_set_1:.*]] = #modeling<index_set {[1,9]}>

// CHECK:       modelica.schedule @schedule {
// CHECK-NEXT:      modelica.main_model {
// CHECK-NEXT:          modelica.schedule_block {
// CHECK-NEXT:              modelica.equation_call @[[eq:.*]] {indices = #modeling<multidim_range [0,8]>}
// CHECK-NEXT:          } {parallelizable = true, readVariables = [#modelica.var<@x, #[[index_set_0]]>], writtenVariables = [#modelica.var<@x, #[[index_set_1]]>]}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:       modelica.equation_function @[[eq]](%[[lb:.*]]: index, %[[ub:.*]]: index) {
// CHECK:           %[[step:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[i0:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:               %[[one:.*]] = arith.constant 1 : index
// CHECK:               %[[i0_remapped:.*]] = arith.addi %[[one]], %[[i0]]
// CHECK:               %[[sub:.*]] = modelica.sub %[[i0_remapped]], %{{.*}}
// CHECK:               modelica.store %{{.*}}[%[[i0_remapped]]], %{{.*}}
// CHECK:           }
// CHECK-NEXT:      modelica.yield
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<10x!modelica.int>

    // x[i] = x[i - 1]
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<10x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<10x!modelica.int>
        %2 = modelica.variable_get @x : !modelica.array<10x!modelica.int>
        %3 = modelica.constant 1 : index
        %4 = modelica.sub %i0, %3 : (index, index) -> index
        %5 = modelica.load %2[%4] : !modelica.array<10x!modelica.int>
        %6 = modelica.equation_side %1 : tuple<!modelica.int>
        %7 = modelica.equation_side %5 : tuple<!modelica.int>
        modelica.equation_sides %6, %7 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.schedule @schedule {
        modelica.main_model {
            modelica.scc {
                modelica.scheduled_equation_instance %t0 {indices = #modeling<multidim_range [1,9]>, iteration_directions = [#modelica<equation_schedule_direction forward>], path = #modelica<equation_path [L, 0]>} : !modelica.equation
            }
        }
    }
}
