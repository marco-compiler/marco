// RUN: modelica-opt %s --split-input-file --schedule-parallelization | FileCheck %s

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
        bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

        bmodelica.schedule @schedule {
            bmodelica.dynamic {
                bmodelica.schedule_block writtenVariables = [@y], readVariables = [] {
                    bmodelica.equation_call @equation_0 {}
                } {parallelizable = true}

                bmodelica.schedule_block writtenVariables = [@x], readVariables = [@y] {
                    bmodelica.equation_call @equation_1 {}
                } {parallelizable = true}

                // CHECK:       bmodelica.parallel_schedule_blocks
                // CHECK-NEXT:      bmodelica.schedule_block writtenVariables = [@y], readVariables = []
                // CHECK-NEXT:          bmodelica.equation_call @equation_0 {}
                // CHECK-NEXT:      }
                // CHECK-SAME:      parallelizable = true

                // CHECK:       bmodelica.parallel_schedule_blocks
                // CHECK-NEXT:      bmodelica.schedule_block writtenVariables = [@x], readVariables = [@y]
                // CHECK-NEXT:          bmodelica.equation_call @equation_1 {}
                // CHECK-NEXT:      }
                // CHECK-SAME:      parallelizable = true
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
