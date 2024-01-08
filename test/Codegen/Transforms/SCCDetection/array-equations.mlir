// RUN: modelica-opt %s --split-input-file --detect-scc --canonicalize-model-for-debug | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
// CHECK:       modelica.main_model {
// CHECK-NEXT:      modelica.scc {
// CHECK-NEXT:          modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,4]>, path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:          modelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:          modelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [3,4]>, path = #modelica<equation_path [L, 0]>}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<5x!modelica.int>
    modelica.variable @y : !modelica.variable<5x!modelica.int>

    // x[i] = y[i]
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<5x!modelica.int>
        %2 = modelica.variable_get @y : !modelica.array<5x!modelica.int>
        %3 = modelica.load %2[%i0] : !modelica.array<5x!modelica.int>
        %4 = modelica.equation_side %1 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // y[i] = 1 - x[i]
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<5x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<5x!modelica.int>
        %2 = modelica.constant #modelica.int<1>
        %3 = modelica.variable_get @x : !modelica.array<5x!modelica.int>
        %4 = modelica.load %3[%i0] : !modelica.array<5x!modelica.int>
        %5 = modelica.sub %2, %4 : (!modelica.int, !modelica.int) -> !modelica.int
        %6 = modelica.equation_side %1 : tuple<!modelica.int>
        %7 = modelica.equation_side %5 : tuple<!modelica.int>
        modelica.equation_sides %6, %7 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // y[i] = 2 - x[i]
    %t2 = modelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = modelica.variable_get @y : !modelica.array<5x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<5x!modelica.int>
        %2 = modelica.constant #modelica.int<2>
        %3 = modelica.variable_get @x : !modelica.array<5x!modelica.int>
        %4 = modelica.load %3[%i0] : !modelica.array<5x!modelica.int>
        %5 = modelica.sub %2, %4 : (!modelica.int, !modelica.int) -> !modelica.int
        %6 = modelica.equation_side %1 : tuple<!modelica.int>
        %7 = modelica.equation_side %5 : tuple<!modelica.int>
        modelica.equation_sides %6, %7 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,4]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t2 {indices = #modeling<multidim_range [3,4]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}
