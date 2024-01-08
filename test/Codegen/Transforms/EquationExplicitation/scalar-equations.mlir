// RUN: modelica-opt %s --split-input-file --equation-explicitation | FileCheck %s

// CHECK:       modelica.schedule @schedule
// CHECK:           modelica.scc_group {
// CHECK-NEXT:          modelica.call @[[eq0:[a-zA-Z_][a-zA-Z0-9_]+]]()
// CHECK-NEXT:      }
// CHECK:           modelica.scc_group {
// CHECK-NEXT:          modelica.call @[[eq1:[a-zA-Z_][a-zA-Z0-9_]+]]()
// CHECK-NEXT:      }

// CHECK:       modelica.equation_function @[[eq0]]() {
// CHECK:           %[[zero:.*]] = modelica.constant #modelica.int<0>
// CHECK:           modelica.simulation_variable_set @y, %[[zero]]
// CHECK:           modelica.yield
// CHECK-NEXT:  }

// CHECK:       modelica.equation_function @[[eq1]]() {
// CHECK:           %[[y:.*]] = modelica.simulation_variable_get @y
// CHECK:           modelica.simulation_variable_set @x, %[[y]]
// CHECK:           modelica.yield
// CHECK-NEXT:  }

module {
    modelica.simulation_variable @x : !modelica.variable<!modelica.int>
    modelica.simulation_variable @y : !modelica.variable<!modelica.int>

    modelica.schedule @schedule {
        // y = 0
        %t0 = modelica.equation_template inductions = [] attributes {id = "t1"} {
            %0 = modelica.simulation_variable_get @y : !modelica.int
            %1 = modelica.constant #modelica.int<0>
            %2 = modelica.equation_side %0 : tuple<!modelica.int>
            %3 = modelica.equation_side %1 : tuple<!modelica.int>
            modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
        }

        // x = y
        %t1 = modelica.equation_template inductions = [] attributes {id = "t0"} {
            %0 = modelica.simulation_variable_get @x : !modelica.int
            %1 = modelica.simulation_variable_get @y : !modelica.int
            %2 = modelica.equation_side %0 : tuple<!modelica.int>
            %3 = modelica.equation_side %1 : tuple<!modelica.int>
            modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
        }

        modelica.scc_group {
            modelica.scc {
                modelica.scheduled_equation_instance %t0 {iteration_directions = [], path = #modelica<equation_path [L, 0]>} : !modelica.equation
            }
        }
        modelica.scc_group {
            modelica.scc {
                modelica.scheduled_equation_instance %t1 {iteration_directions = [], path = #modelica<equation_path [L, 0]>} : !modelica.equation
            }
        }
    }
}
