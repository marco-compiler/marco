// RUN: modelica-opt %s --split-input-file --schedule-parallelization | FileCheck %s

// CHECK-DAG: #[[index_set_0:.*]] = #modeling<index_set {[0,3]}>
// CHECK-DAG: #[[index_set_1:.*]] = #modeling<index_set {[0,5]}>
// CHECK-DAG: #[[index_set_2:.*]] = #modeling<index_set {[4,9]}>

// CHECK:       modelica.schedule @schedule {
// CHECK-NEXT:      modelica.main_model {
// CHECK-NEXT:          modelica.parallel_schedule_blocks {
// CHECK-NEXT:              modelica.schedule_block {
// CHECK-NEXT:                  modelica.equation_call @equation_0
// CHECK-NEXT:              } {parallelizable = true, readVariables = [], writtenVariables = [#modelica.var<@x, #[[index_set_0]]>]}
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.parallel_schedule_blocks {
// CHECK-NEXT:              modelica.schedule_block {
// CHECK-NEXT:                  modelica.equation_call @equation_1
// CHECK-NEXT:              } {parallelizable = true, readVariables = [#modelica.var<@x, #[[index_set_1]]>], writtenVariables = [#modelica.var<@x, #[[index_set_2]]>]}
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
    modelica.model @Test {
        modelica.variable @x : !modelica.variable<10x!modelica.int>

        modelica.schedule @schedule {
            modelica.main_model {
                modelica.schedule_block {
                    modelica.equation_call @equation_0
                } {parallelizable = true, readVariables = [], writtenVariables = [#modelica.var<@x, #modeling<index_set {[0,3]}>>]}
                modelica.schedule_block {
                    modelica.equation_call @equation_1
                } {parallelizable = true, readVariables = [#modelica.var<@x, #modeling<index_set {[0,5]}>>], writtenVariables = [#modelica.var<@x, #modeling<index_set {[4,9]}>>]}
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
