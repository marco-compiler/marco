// RUN: modelica-opt %s --split-input-file --equation-explicitation | FileCheck %s

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.dynamic {
// CHECK-NEXT:          bmodelica.schedule_block writtenVariables = [@y], readVariables = [] {
// CHECK-NEXT:              bmodelica.equation_call @[[eq0:.*]]
// CHECK-NEXT:          } {parallelizable = true}
// CHECK-NEXT:          bmodelica.schedule_block writtenVariables = [@x], readVariables = [@y] {
// CHECK-NEXT:              bmodelica.equation_call @[[eq1:.*]]
// CHECK-NEXT:          } {parallelizable = true}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:       bmodelica.equation_function @[[eq0]]() {
// CHECK:           %[[zero:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK:           bmodelica.qualified_variable_set @Test::@y, %[[zero]]
// CHECK:           bmodelica.yield
// CHECK-NEXT:  }

// CHECK:       bmodelica.equation_function @[[eq1]]() {
// CHECK:           %[[y:.*]] = bmodelica.qualified_variable_get @Test::@y
// CHECK:           bmodelica.qualified_variable_set @Test::@x, %[[y]]
// CHECK:           bmodelica.yield
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    // y = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // x = y
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.variable_get @y : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.schedule @schedule {
        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.scheduled_equation_instance %t0 {iteration_directions = [], path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
            }
            bmodelica.scc {
                bmodelica.scheduled_equation_instance %t1 {iteration_directions = [], path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
            }
        }
    }
}
