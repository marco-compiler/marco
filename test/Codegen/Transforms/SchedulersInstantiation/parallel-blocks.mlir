// RUN: modelica-opt %s --split-input-file --schedulers-instantiation | FileCheck %s

// CHECK:       modelica.schedule @schedule {
// CHECK-NEXT:      modelica.main_model {
// CHECK-NEXT:          modelica.parallel_schedule_blocks {
// CHECK-NEXT:              modelica.schedule_block {
// CHECK-NEXT:                  simulation.scheduler_run @[[scheduler:.*]]
// CHECK-NEXT:              } {readVariables = [], writtenVariables = [#modelica.var<@x>, #modelica.var<@y>]}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:       simulation.scheduler @[[scheduler]]

// CHECK:       simulation.dynamic_model_begin {
// CHECK-NEXT:      simulation.scheduler_create @[[scheduler]]
// CHECK-NEXT:      simulation.scheduler_add_equation @[[scheduler]] {function = @[[equation_0_wrapper:.*]]}
// CHECK-NEXT:      simulation.scheduler_add_equation @[[scheduler]] {function = @[[equation_1_wrapper:.*]]}
// CHECK-NEXT:  }

// CHECK:       simulation.equation_function @[[equation_0_wrapper]]() {
// CHECK-NEXT:      modelica.call @equation_0()
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

// CHECK:       simulation.equation_function @[[equation_1_wrapper]]() {
// CHECK-NEXT:      modelica.call @equation_1()
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

// CHECK:       simulation.dynamic_model_end {
// CHECK-NEXT:      simulation.scheduler_destroy @[[scheduler]]
// CHECK-NEXT:  }

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.int>

        modelica.schedule @schedule {
            modelica.main_model {
                modelica.parallel_schedule_blocks {
                    modelica.schedule_block {
                        modelica.equation_call @equation_0
                    } {parallelizable = true, readVariables = [], writtenVariables = [#modelica.var<@x>]}
                    modelica.schedule_block {
                        modelica.equation_call @equation_1
                    } {parallelizable = true, readVariables = [], writtenVariables = [#modelica.var<@y>]}
                }
            }
        }
    }

    modelica.equation_function @equation_0() {
        modelica.yield
    }

    modelica.equation_function @equation_1() {
        modelica.yield
    }
}
