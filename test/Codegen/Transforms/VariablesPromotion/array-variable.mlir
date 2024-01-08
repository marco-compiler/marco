// RUN: modelica-opt %s --split-input-file --promote-variables-to-parameters --canonicalize | FileCheck %s

// Variable depending on a constant.

// CHECK-DAG: modelica.variable @x : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK: modelica.initial_model
// CHECK: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    // x = 0
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.constant #modelica.int<0>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}

// -----

// Variable depending on a parameter.

// CHECK-DAG: modelica.variable @x : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: modelica.variable @y : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK: modelica.initial_model
// CHECK: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK: modelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int, parameter>
    modelica.variable @y : !modelica.variable<3x!modelica.int>

    // x[i] = 0
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.constant #modelica.int<0>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // y[i] = x[i]
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %3 = modelica.load %2[%i0] : !modelica.array<3x!modelica.int>
        %4 = modelica.equation_side %1 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.initial_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }

    modelica.main_model {
        modelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}

// -----

// Variable depending on time.

// CHECK-DAG: modelica.variable @x : !modelica.variable<3x!modelica.int>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK: modelica.main_model
// CHECK: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    // x[i] = time
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.time : !modelica.real
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}

// -----

// Variable z depending on the non-parameter variable y.
// Variable y depending on the parameter x.

// CHECK-DAG: modelica.variable @x : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: modelica.variable @y : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: modelica.variable @z : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
// CHECK: modelica.initial_model
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int, parameter>
    modelica.variable @y : !modelica.variable<3x!modelica.int>
    modelica.variable @z : !modelica.variable<3x!modelica.int>

    // x[i] = 0
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.constant #modelica.int<0>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // y[i] = x[i]
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %3 = modelica.load %2[%i0] : !modelica.array<3x!modelica.int>
        %4 = modelica.equation_side %1 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // z[i] = y[i]
    %t2 = modelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = modelica.variable_get @z : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %3 = modelica.load %2[%i0] : !modelica.array<3x!modelica.int>
        %4 = modelica.equation_side %1 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.initial_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }

    modelica.main_model {
        modelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t2 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}

// -----

// Promotable SCC.

// CHECK-DAG: modelica.variable @x : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: modelica.variable @y : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK: modelica.initial_model
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK-NOT: modelica.main_model

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>
    modelica.variable @y : !modelica.variable<3x!modelica.int>

    // x[i] = y[i]
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %3 = modelica.load %2[%i0] : !modelica.array<3x!modelica.int>
        %4 = modelica.equation_side %1 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // y[i] = x[i]
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %3 = modelica.load %2[%i0] : !modelica.array<3x!modelica.int>
        %4 = modelica.equation_side %1 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}

// -----

// Promotable SCC depending on a promotable variable.

// CHECK-DAG: modelica.variable @x : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: modelica.variable @y : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: modelica.variable @z : !modelica.variable<3x!modelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
// CHECK: modelica.initial_model
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}
// CHECK-NOT: modelica.main_model

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>
    modelica.variable @y : !modelica.variable<3x!modelica.int>
    modelica.variable @z : !modelica.variable<3x!modelica.int>

    // x[i] = 0
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.constant #modelica.int<0>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // y[i] = x[i] + z[i]
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %3 = modelica.load %2[%i0] : !modelica.array<3x!modelica.int>
        %4 = modelica.variable_get @z : !modelica.array<3x!modelica.int>
        %5 = modelica.load %4[%i0] : !modelica.array<3x!modelica.int>
        %6 = modelica.add %3, %5 : (!modelica.int, !modelica.int) -> !modelica.int
        %7 = modelica.equation_side %1 : tuple<!modelica.int>
        %8 = modelica.equation_side %6 : tuple<!modelica.int>
        modelica.equation_sides %7, %8 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    // z[i] = x[i] + y[i]
    %t2 = modelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = modelica.variable_get @z : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %3 = modelica.load %2[%i0] : !modelica.array<3x!modelica.int>
        %4 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %5 = modelica.load %4[%i0] : !modelica.array<3x!modelica.int>
        %6 = modelica.add %3, %5 : (!modelica.int, !modelica.int) -> !modelica.int
        %7 = modelica.equation_side %1 : tuple<!modelica.int>
        %8 = modelica.equation_side %6 : tuple<!modelica.int>
        modelica.equation_sides %7, %8 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t2 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}

// -----

// Variable depending on a variable that is not written by any other equation
// (and, thus, potentially a state variable).

// CHECK-DAG: modelica.variable @x : !modelica.variable<3x!modelica.int>
// CHECK-DAG: modelica.variable @y : !modelica.variable<3x!modelica.int>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK: modelica.main_model
// CHECK: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>
    modelica.variable @y : !modelica.variable<3x!modelica.int>

    // y[i] = x[i]
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
        %1 = modelica.load %0[%i0] : !modelica.array<3x!modelica.int>
        %2 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        %3 = modelica.load %2[%i0] : !modelica.array<3x!modelica.int>
        %4 = modelica.equation_side %1 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}

// -----

// Promotable array written by different equations.

// CHECK-DAG: modelica.variable @x : !modelica.variable<2x!modelica.real, parameter>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK: modelica.initial_model
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,0]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [1,1]>, path = #modelica<equation_path [L, 0]>}
// CHECK-NOT: modelica.main_model

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.real>

    // x[i] = 1
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<2x!modelica.real>
        %2 = modelica.constant #modelica.real<0.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // x[i] = 1
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<2x!modelica.real>
        %2 = modelica.constant #modelica.real<1.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,0]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [1,1]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}

// -----

// Array not fully promotable.

// CHECK-DAG: modelica.variable @x : !modelica.variable<2x!modelica.real>
// CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK: modelica.main_model
// CHECK-DAG: modelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,0]>, path = #modelica<equation_path [L, 0]>}
// CHECK-DAG: modelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [1,1]>, path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.real>

    // x[i] = 0
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<2x!modelica.real>
        %2 = modelica.constant #modelica.real<0.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // x[i] = time
    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<2x!modelica.real>
        %2 = modelica.time : !modelica.real
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,0]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [1,1]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}
