// RUN: modelica-opt %s --split-input-file --schedulers-instantiation | FileCheck %s

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.dynamic {
// CHECK-NEXT:          bmodelica.parallel_schedule_blocks {
// CHECK-NEXT:              bmodelica.schedule_block writtenVariables = [@x, @y], readVariables = [] {
// CHECK-NEXT:                  runtime.scheduler_run @[[scheduler:.*]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:       runtime.scheduler @[[scheduler]]

// CHECK:       runtime.dynamic_model_begin {
// CHECK-NEXT:      runtime.scheduler_create @[[scheduler]]
// CHECK-NEXT:      runtime.scheduler_add_equation @[[scheduler]], @[[equation_0_wrapper:.*]] {}
// CHECK-NEXT:      runtime.scheduler_add_equation @[[scheduler]], @[[equation_1_wrapper:.*]] {}
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
            bmodelica.dynamic {
                bmodelica.parallel_schedule_blocks {
                    bmodelica.schedule_block writtenVariables = [@x], readVariables = [] {
                        bmodelica.equation_call @equation_0 {}
                    } {parallelizable = true}
                    bmodelica.schedule_block writtenVariables = [@y], readVariables = [] {
                        bmodelica.equation_call @equation_1 {}
                    } {parallelizable = true}
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
