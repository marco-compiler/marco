// RUN: modelica-opt %s --split-input-file --schedule-parallelization | FileCheck %s

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.dynamic {
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block writtenVariables = [<@x, {[0,3]}>], readVariables = [] {
// CHECK-NEXT:                  bmodelica.equation_call @equation_0
// CHECK-NEXT:              } {parallelizable = true}
// CHECK-NEXT:          }
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block writtenVariables = [<@x, {[4,9]}>], readVariables = [<@x, {[0,5]}>] {
// CHECK-NEXT:                  bmodelica.equation_call @equation_1
// CHECK-NEXT:              } {parallelizable = true}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.int>

        bmodelica.schedule @schedule {
            bmodelica.dynamic {
                bmodelica.schedule_block writtenVariables = [<@x, {[0,3]}>], readVariables = [] {
                    bmodelica.equation_call @equation_0
                } {parallelizable = true}
                bmodelica.schedule_block writtenVariables = [<@x, {[4,9]}>], readVariables = [<@x, {[0,5]}>] {
                    bmodelica.equation_call @equation_1
                } {parallelizable = true}
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
