// RUN: modelica-opt %s --split-input-file --schedule-parallelization | FileCheck %s

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.main_model {
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block {
// CHECK-NEXT:                  bmodelica.equation_call @equation_0
// CHECK-NEXT:              } {parallelizable = true, readVariables = [], writtenVariables = [#bmodelica.var<@y>]}
// CHECK-NEXT:          }
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block {
// CHECK-NEXT:                  bmodelica.equation_call @equation_1
// CHECK-NEXT:              } {parallelizable = true, readVariables = [#bmodelica.var<@y>], writtenVariables = [#bmodelica.var<@x>]}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
        bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

        bmodelica.schedule @schedule {
            bmodelica.main_model {
                bmodelica.schedule_block {
                    bmodelica.equation_call @equation_0
                } {parallelizable = true, readVariables = [], writtenVariables = [#bmodelica.var<@y>]}

                bmodelica.schedule_block {
                    bmodelica.equation_call @equation_1
                } {parallelizable = true, readVariables = [#bmodelica.var<@y>], writtenVariables = [#bmodelica.var<@x>]}
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
