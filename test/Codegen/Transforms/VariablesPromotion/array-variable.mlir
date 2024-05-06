// RUN: modelica-opt %s --split-input-file --promote-variables-to-parameters --canonicalize | FileCheck %s

// Variable depending on a constant.

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK: bmodelica.initial
// CHECK: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    // x = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica.int<0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}

// -----

// Variable depending on a parameter.

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK: bmodelica.initial
// CHECK: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>

    // x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica.int<0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y[i] = x[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %3 = bmodelica.load %2[%i0] : !bmodelica.array<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.initial {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}

// -----

// Variable depending on time.

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK: bmodelica.dynamic
// CHECK: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    // x[i] = time
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.time : !bmodelica.real
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}

// -----

// Variable z depending on the non-parameter variable y.
// Variable y depending on the parameter x.

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: bmodelica.variable @z : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
// CHECK: bmodelica.initial
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @z : !bmodelica.variable<3x!bmodelica.int>

    // x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica.int<0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y[i] = x[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %3 = bmodelica.load %2[%i0] : !bmodelica.array<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // z[i] = y[i]
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
        %3 = bmodelica.load %2[%i0] : !bmodelica.array<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.initial {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t2 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}

// -----

// Promotable SCC.

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK: bmodelica.initial
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-NOT: bmodelica.dynamic

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>

    // x[i] = y[i]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
        %3 = bmodelica.load %2[%i0] : !bmodelica.array<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y[i] = x[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %3 = bmodelica.load %2[%i0] : !bmodelica.array<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}

// -----

// Promotable SCC depending on a promotable variable.

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: bmodelica.variable @z : !bmodelica.variable<3x!bmodelica.int, parameter>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}
// CHECK: bmodelica.initial
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-NOT: bmodelica.dynamic

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @z : !bmodelica.variable<3x!bmodelica.int>

    // x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica.int<0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y[i] = x[i] + z[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %3 = bmodelica.load %2[%i0] : !bmodelica.array<3x!bmodelica.int>
        %4 = bmodelica.variable_get @z : !bmodelica.array<3x!bmodelica.int>
        %5 = bmodelica.load %4[%i0] : !bmodelica.array<3x!bmodelica.int>
        %6 = bmodelica.add %3, %5 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %7 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.int>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // z[i] = x[i] + y[i]
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %3 = bmodelica.load %2[%i0] : !bmodelica.array<3x!bmodelica.int>
        %4 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
        %5 = bmodelica.load %4[%i0] : !bmodelica.array<3x!bmodelica.int>
        %6 = bmodelica.add %3, %5 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %7 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.int>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t2 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}

// -----

// Variable depending on a variable that is not written by any other equation
// (and, thus, potentially a state variable).

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
// CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK: bmodelica.dynamic
// CHECK: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>

    // y[i] = x[i]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
        %3 = bmodelica.load %2[%i0] : !bmodelica.array<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}

// -----

// Promotable array written by different equations.

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real, parameter>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK: bmodelica.initial
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,0]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [1,1]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-NOT: bmodelica.dynamic

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    // x[i] = 1
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<2x!bmodelica.real>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica.real<0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // x[i] = 1
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<2x!bmodelica.real>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica.real<1.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,0]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [1,1]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}

// -----

// Array not fully promotable.

// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}
// CHECK: bmodelica.dynamic
// CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,0]>, path = #bmodelica<equation_path [L, 0]>}
// CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [1,1]>, path = #bmodelica<equation_path [L, 0]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    // x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<2x!bmodelica.real>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica.real<0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // x[i] = time
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<2x!bmodelica.real>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<2x!bmodelica.real>
        %2 = bmodelica.time : !bmodelica.real
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,0]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [1,1]>, path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}
