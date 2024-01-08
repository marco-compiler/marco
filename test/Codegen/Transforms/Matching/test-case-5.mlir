// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:4
//   x[i] = 10
// for i in 0:3
//   y[i] = x[i + 1]
// for i in 0:3
//   z[i] = x[i] + y[i]
// z[4] = x[4]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<5x!modelica.real>
    modelica.variable @y : !modelica.variable<4x!modelica.real>
    modelica.variable @z : !modelica.variable<5x!modelica.real>

    // [i] = 10
    // CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<5x!modelica.real>
        %2 = modelica.constant #modelica.real<10.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // y[i] = x[i + 1]
    // CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<4x!modelica.real>
        %1 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<4x!modelica.real>
        %3 = modelica.constant 1 : index
        %4 = modelica.add %i0, %3 : (index, index) -> index
        %5 = modelica.load %1[%4] : !modelica.array<5x!modelica.real>
        %6 = modelica.equation_side %2 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // z[i] = x[i] + y[i]
    // CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
    %t2 = modelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = modelica.variable_get @z : !modelica.array<5x!modelica.real>
        %1 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %2 = modelica.variable_get @y : !modelica.array<4x!modelica.real>
        %3 = modelica.load %0[%i0] : !modelica.array<5x!modelica.real>
        %4 = modelica.load %1[%i0] : !modelica.array<5x!modelica.real>
        %5 = modelica.load %2[%i0] : !modelica.array<4x!modelica.real>
        %6 = modelica.add %4, %5 : (!modelica.real, !modelica.real) -> !modelica.real
        %7 = modelica.equation_side %3 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // z[4] = x[4]
    // CHECK-DAG: %[[t3:.*]] = modelica.equation_template inductions = [] attributes {id = "t3"}
    %t3 = modelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = modelica.variable_get @z : !modelica.array<5x!modelica.real>
        %1 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %2 = modelica.constant 4 : index
        %3 = modelica.load %0[%2] : !modelica.array<5x!modelica.real>
        %4 = modelica.load %1[%2] : !modelica.array<5x!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        // CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,4]>, path = #modelica<equation_path [L, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,3]>, path = #modelica<equation_path [L, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [0,3]>, path = #modelica<equation_path [L, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t3]] {path = #modelica<equation_path [L, 0]>}
        modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,4]>} : !modelica.equation
        modelica.equation_instance %t1 {indices = #modeling<multidim_range [0,3]>} : !modelica.equation
        modelica.equation_instance %t2 {indices = #modeling<multidim_range [0,3]>} : !modelica.equation
        modelica.equation_instance %t3 : !modelica.equation
    }
}
