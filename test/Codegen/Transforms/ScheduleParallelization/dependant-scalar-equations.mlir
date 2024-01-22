// RUN: modelica-opt %s --split-input-file --schedule-parallelization | FileCheck %s

// CHECK:       modelica.schedule @schedule {
// CHECK-NEXT:      modelica.main_model {
// CHECK-NEXT:          modelica.parallel_schedule_blocks {
// CHECK-NEXT:              modelica.schedule_block {
// CHECK-NEXT:                  modelica.equation_call @equation_0
// CHECK-NEXT:              } {parallelizable = true, readVariables = [], writtenVariables = [#modelica.var<@y>]}
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.parallel_schedule_blocks {
// CHECK-NEXT:              modelica.schedule_block {
// CHECK-NEXT:                  modelica.equation_call @equation_1
// CHECK-NEXT:              } {parallelizable = true, readVariables = [#modelica.var<@y>], writtenVariables = [#modelica.var<@x>]}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<!modelica.int>
        modelica.variable @y : !modelica.variable<!modelica.int>

        modelica.schedule @schedule {
            modelica.main_model {
                modelica.schedule_block {
                    modelica.equation_call @equation_0
                } {parallelizable = true, readVariables = [], writtenVariables = [#modelica.var<@y>]}

                modelica.schedule_block {
                    modelica.equation_call @equation_1
                } {parallelizable = true, readVariables = [#modelica.var<@y>], writtenVariables = [#modelica.var<@x>]}
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
