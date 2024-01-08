// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// l + fl = 0
// fl = 0
// h + fh = 0
// fh = 0
// for i in 0:4
//   fl + f[i] + x[i] = 0
// for i in 0:4
//   fh + f[i] + y[i] = 0
// for i in 0:4
//   f[i] = 0

modelica.model @Test {
    modelica.variable @l : !modelica.variable<!modelica.real>
    modelica.variable @h : !modelica.variable<!modelica.real>
    modelica.variable @fl : !modelica.variable<!modelica.real>
    modelica.variable @fh : !modelica.variable<!modelica.real>
    modelica.variable @x : !modelica.variable<5x!modelica.real>
    modelica.variable @y : !modelica.variable<5x!modelica.real>
    modelica.variable @f : !modelica.variable<5x!modelica.real>

    // l + fl = 0
    // CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @l : !modelica.real
        %1 = modelica.variable_get @fl : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        %3 = modelica.constant #modelica.real<0.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // fl = 0
    // CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @fl : !modelica.real
        %1 = modelica.constant #modelica.real<0.0>
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // h + fh = 0
    // CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = modelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = modelica.variable_get @h : !modelica.real
        %1 = modelica.variable_get @fh : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        %3 = modelica.constant #modelica.real<0.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // fh = 0
    // CHECK-DAG: %[[t3:.*]] = modelica.equation_template inductions = [] attributes {id = "t3"}
    %t3 = modelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = modelica.variable_get @fh : !modelica.real
        %1 = modelica.constant #modelica.real<0.0>
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // fl + f[i] + x[i] = 0
    // CHECK-DAG: %[[t4:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t4"}
    %t4 = modelica.equation_template inductions = [%i0] attributes {id = "t4"} {
        %0 = modelica.variable_get @fl : !modelica.real
        %1 = modelica.variable_get @f : !modelica.array<5x!modelica.real>
        %2 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %3 = modelica.load %1[%i0] : !modelica.array<5x!modelica.real>
        %4 = modelica.load %2[%i0] : !modelica.array<5x!modelica.real>
        %5 = modelica.add %0, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.add %5, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %7 = modelica.constant #modelica.real<0.0>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        %9 = modelica.equation_side %7 : tuple<!modelica.real>
        modelica.equation_sides %8, %9 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // fh + f[i] + y[i] = 0
    // CHECK-DAG: %[[t5:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t5"}
    %t5 = modelica.equation_template inductions = [%i0] attributes {id = "t5"} {
        %0 = modelica.variable_get @fh : !modelica.real
        %1 = modelica.variable_get @f : !modelica.array<5x!modelica.real>
        %2 = modelica.variable_get @y : !modelica.array<5x!modelica.real>
        %3 = modelica.load %1[%i0] : !modelica.array<5x!modelica.real>
        %4 = modelica.load %2[%i0] : !modelica.array<5x!modelica.real>
        %5 = modelica.add %0, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.add %5, %4 : (!modelica.real, !modelica.real) -> !modelica.real
        %7 = modelica.constant #modelica.real<0.0>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        %9 = modelica.equation_side %7 : tuple<!modelica.real>
        modelica.equation_sides %8, %9 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // f[i] = 0
    // CHECK-DAG: %[[t6:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t6"}
    %t6 = modelica.equation_template inductions = [%i0] attributes {id = "t6"} {
        %0 = modelica.variable_get @f : !modelica.array<5x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<5x!modelica.real>
        %2 = modelica.constant #modelica.real<0.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        // CHECK-DAG: modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [L, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t2]] {path = #modelica<equation_path [L, 0, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t3]] {path = #modelica<equation_path [L, 0]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t4]] {indices = #modeling<multidim_range [0,4]>, path = #modelica<equation_path [L, 0, 1]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t5]] {indices = #modeling<multidim_range [0,4]>, path = #modelica<equation_path [L, 0, 1]>}
        // CHECK-DAG: modelica.matched_equation_instance %[[t6]] {indices = #modeling<multidim_range [0,4]>, path = #modelica<equation_path [L, 0]>}
        modelica.equation_instance %t0 : !modelica.equation
        modelica.equation_instance %t1 : !modelica.equation
        modelica.equation_instance %t2 : !modelica.equation
        modelica.equation_instance %t3 : !modelica.equation
        modelica.equation_instance %t4 {indices = #modeling<multidim_range [0,4]>} : !modelica.equation
        modelica.equation_instance %t5 {indices = #modeling<multidim_range [0,4]>} : !modelica.equation
        modelica.equation_instance %t6 {indices = #modeling<multidim_range [0,4]>} : !modelica.equation
    }
}
