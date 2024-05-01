// RUN: modelica-opt %s --split-input-file --schedule-parallelization | FileCheck %s

// CHECK-DAG: #[[index_set_0:.*]] = #modeling<index_set {[0,3]}>
// CHECK-DAG: #[[index_set_1:.*]] = #modeling<index_set {[0,5]}>
// CHECK-DAG: #[[index_set_2:.*]] = #modeling<index_set {[4,9]}>

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.main_model {
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block {
// CHECK-NEXT:                  bmodelica.equation_call @equation_0
// CHECK-NEXT:              } {parallelizable = true, readVariables = [], writtenVariables = [#bmodelica.var<@x, #[[index_set_0]]>]}
// CHECK-NEXT:          }
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block {
// CHECK-NEXT:                  bmodelica.equation_call @equation_1
// CHECK-NEXT:              } {parallelizable = true, readVariables = [#bmodelica.var<@x, #[[index_set_1]]>], writtenVariables = [#bmodelica.var<@x, #[[index_set_2]]>]}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.int>

        bmodelica.schedule @schedule {
            bmodelica.main_model {
                bmodelica.schedule_block {
                    bmodelica.equation_call @equation_0
                } {parallelizable = true, readVariables = [], writtenVariables = [#bmodelica.var<@x, #modeling<index_set {[0,3]}>>]}
                bmodelica.schedule_block {
                    bmodelica.equation_call @equation_1
                } {parallelizable = true, readVariables = [#bmodelica.var<@x, #modeling<index_set {[0,5]}>>], writtenVariables = [#bmodelica.var<@x, #modeling<index_set {[4,9]}>>]}
            }
        }
    }

    bmodelica.equation_function @equation_0() {
        bmodelica.yield
    }

    bmodelica.equation_function @equation_1() {
        bmodelica.yield
    }
}
