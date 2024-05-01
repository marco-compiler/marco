// RUN: modelica-opt %s --split-input-file --schedulers-instantiation | FileCheck %s

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.main_model {
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block {
// CHECK-NEXT:                  runtime.scheduler_run @[[scheduler:.*]]
// CHECK-NEXT:              } {readVariables = [], writtenVariables = [#bmodelica.var<@x>, #bmodelica.var<@y>]}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:       runtime.scheduler @[[scheduler]]

// CHECK:       runtime.dynamic_model_begin {
// CHECK-NEXT:      runtime.scheduler_create @[[scheduler]]
// CHECK-NEXT:      runtime.scheduler_add_equation @[[scheduler]] {function = @[[equation_0_wrapper:.*]]}
// CHECK-NEXT:      runtime.scheduler_add_equation @[[scheduler]] {function = @[[equation_1_wrapper:.*]]}
// CHECK-NEXT:  }

// CHECK:       runtime.equation_function @[[equation_0_wrapper]]() {
// CHECK-NEXT:      bmodelica.call @equation_0()
// CHECK-NEXT:      runtime.return
// CHECK-NEXT:  }

// CHECK:       runtime.equation_function @[[equation_1_wrapper]]() {
// CHECK-NEXT:      bmodelica.call @equation_1()
// CHECK-NEXT:      runtime.return
// CHECK-NEXT:  }

// CHECK:       runtime.dynamic_model_end {
// CHECK-NEXT:      runtime.scheduler_destroy @[[scheduler]]
// CHECK-NEXT:  }

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

        bmodelica.schedule @schedule {
            bmodelica.main_model {
                bmodelica.parallel_schedule_blocks {
                    bmodelica.schedule_block {
                        bmodelica.equation_call @equation_0
                    } {parallelizable = true, readVariables = [], writtenVariables = [#bmodelica.var<@x>]}
                    bmodelica.schedule_block {
                        bmodelica.equation_call @equation_1
                    } {parallelizable = true, readVariables = [], writtenVariables = [#bmodelica.var<@y>]}
                }
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
