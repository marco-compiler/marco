// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// i = 1 to 8
//   x[i - 1] = 3 - x[4];
//
// x[9] = 0

// CHECK: %[[t0:.*]] = modelica.equation_template inductions = [{{.*}}] attributes {id = "t0"}

// CHECK:       modelica.scc {
// CHECK-NEXT:      modelica.scheduled_equation_instance %[[t0]] {indices = #modeling<multidim_range [5,5]>, iteration_directions = [#modelica<equation_schedule_direction forward>], path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:  }
// CHECK:       modelica.scc {
// CHECK-NEXT:      modelica.scheduled_equation_instance %[[t0]] {indices = #modeling<multidim_range [1,4]>, iteration_directions = [#modelica<equation_schedule_direction forward>], path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:  }
// CHECK:       modelica.scc {
// CHECK-NEXT:      modelica.scheduled_equation_instance %[[t0]] {indices = #modeling<multidim_range [6,8]>, iteration_directions = [#modelica<equation_schedule_direction forward>], path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<10x!modelica.real>

    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
      %0 = modelica.variable_get @x : !modelica.array<10x!modelica.real>
      %1 = modelica.constant 1 : index
      %2 = modelica.sub %i0, %1 : (index, index) -> index
      %3 = modelica.load %0[%2] : !modelica.array<10x!modelica.real>
      %4 = modelica.constant #modelica.real<3.0> : !modelica.real
      %5 = modelica.constant 4 : index
      %6 = modelica.load %0[%5] : !modelica.array<10x!modelica.real>
      %7 = modelica.sub %4, %6 : (!modelica.real, !modelica.real) -> !modelica.real
      %8 = modelica.equation_side %3 : tuple<!modelica.real>
      %9 = modelica.equation_side %7 : tuple<!modelica.real>
      modelica.equation_sides %8, %9 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [1,8]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation

    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
      %0 = modelica.variable_get @x : !modelica.array<10x!modelica.real>
      %1 = modelica.constant 9 : index
      %2 = modelica.load %0[%1] : !modelica.array<10x!modelica.real>
      %3 = modelica.constant #modelica.real<0.0> : !modelica.real
      %4 = modelica.equation_side %2 : tuple<!modelica.real>
      %5 = modelica.equation_side %3 : tuple<!modelica.real>
      modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.matched_equation_instance %t1 {path = #modelica<equation_path [L, 0]>} : !modelica.equation
}
