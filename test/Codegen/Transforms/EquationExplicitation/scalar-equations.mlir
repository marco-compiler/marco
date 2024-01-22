// RUN: modelica-opt %s --split-input-file --equation-explicitation | FileCheck %s

// CHECK:       modelica.schedule @schedule {
// CHECK-NEXT:      modelica.main_model {
// CHECK-NEXT:          modelica.schedule_block {
// CHECK-NEXT:              modelica.equation_call @[[eq0:.*]]
// CHECK-NEXT:          } {parallelizable = true, readVariables = [], writtenVariables = [#modelica.var<@y>]}
// CHECK-NEXT:          modelica.schedule_block {
// CHECK-NEXT:              modelica.equation_call @[[eq1:.*]]
// CHECK-NEXT:          } {parallelizable = true, readVariables = [#modelica.var<@y>], writtenVariables = [#modelica.var<@x>]}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:       modelica.equation_function @[[eq0]]() {
// CHECK:           %[[zero:.*]] = modelica.constant #modelica.int<0>
// CHECK:           modelica.qualified_variable_set @Test::@y, %[[zero]]
// CHECK:           modelica.yield
// CHECK-NEXT:  }

// CHECK:       modelica.equation_function @[[eq1]]() {
// CHECK:           %[[y:.*]] = modelica.qualified_variable_get @Test::@y
// CHECK:           modelica.qualified_variable_set @Test::@x, %[[y]]
// CHECK:           modelica.yield
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    // y = 0
    %t0 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // x = y
    %t1 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.schedule @schedule {
        modelica.main_model {
            modelica.scc {
                modelica.scheduled_equation_instance %t0 {iteration_directions = [], path = #modelica<equation_path [L, 0]>} : !modelica.equation
            }
            modelica.scc {
                modelica.scheduled_equation_instance %t1 {iteration_directions = [], path = #modelica<equation_path [L, 0]>} : !modelica.equation
            }
        }
    }
}
