// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// for i in 0:9 loop
//    x[i] = x[i + 1]
//
// x[9] = 0

// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK:       modelica.scc_group {
// CHECK-NEXT:      modelica.scc {
// CHECK-NEXT:          modelica.scheduled_equation_instance %[[t1]] {iteration_directions = [], path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK-NEXT:  modelica.scc_group {
// CHECK-NEXT:      modelica.scc {
// CHECK-NEXT:          modelica.scheduled_equation_instance %[[t0]] {indices = #modeling<multidim_range [1,8]>, iteration_directions = [#modelica<equation_schedule_direction backward>], path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK-NEXT:  modelica.scc_group {
// CHECK-NEXT:      modelica.scc {
// CHECK-NEXT:          modelica.scheduled_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,0]>, iteration_directions = [#modelica<equation_schedule_direction forward>], path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<10x!modelica.int>

    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<10x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<10x!modelica.int>
        %2 = modelica.variable_get @x : !modelica.array<10x!modelica.int>
        %3 = modelica.constant 1 : index
        %4 = modelica.add %i0, %3 : (index, index) -> index
        %5 = modelica.load %2[%4] : !modelica.array<10x!modelica.int>
        %6 = modelica.equation_side %1 : tuple<!modelica.int>
        %7 = modelica.equation_side %5 : tuple<!modelica.int>
        modelica.equation_sides %6, %7 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,8]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation

    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<10x!modelica.int>
        %1 = modelica.constant 9 : index
        %2 = modelica.load %0[%1] : !modelica.array<10x!modelica.int>
        %3 = modelica.constant #modelica.int<0>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}
