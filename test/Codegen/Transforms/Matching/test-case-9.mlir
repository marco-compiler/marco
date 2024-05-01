// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:4
//   x[i] - y[i] = 0
// x[0] + x[1] + x[2] + x[3] + x[4] = 2
// y[0] + y[1] + y[2] + y[3] + y[4] = 3
// x[0] - x[1] + x[2] + x[3] + x[4] = 2
// y[0] + y[1] - y[2] + y[3] + y[4] = 3
// x[0] + x[1] + x[2] - x[3] + x[4] = 2

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<5x!bmodelica.real>

    // x[i] - y[i] = 0
    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.variable_get @y : !bmodelica.array<5x!bmodelica.real>
        %2 = bmodelica.load %0[%i0] : !bmodelica.array<5x!bmodelica.real>
        %3 = bmodelica.load %1[%i0] : !bmodelica.array<5x!bmodelica.real>
        %4 = bmodelica.sub %2, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %5 = bmodelica.constant #bmodelica.real<0.0>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // x[0] + x[1] + x[2] + x[3] + x[4] = 2
    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.constant 3 : index
        %5 = bmodelica.constant 4 : index
        %6 = bmodelica.load %0[%1] : !bmodelica.array<5x!bmodelica.real>
        %7 = bmodelica.load %0[%2] : !bmodelica.array<5x!bmodelica.real>
        %8 = bmodelica.load %0[%3] : !bmodelica.array<5x!bmodelica.real>
        %9 = bmodelica.load %0[%4] : !bmodelica.array<5x!bmodelica.real>
        %10 = bmodelica.load %0[%5] : !bmodelica.array<5x!bmodelica.real>
        %11 = bmodelica.add %6, %7 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %12 = bmodelica.add %11, %8 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %13 = bmodelica.add %12, %9 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %14 = bmodelica.add %13, %10 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %15 = bmodelica.constant #bmodelica.real<2.0>
        %16 = bmodelica.equation_side %14 : tuple<!bmodelica.real>
        %17 = bmodelica.equation_side %15 : tuple<!bmodelica.real>
        bmodelica.equation_sides %16, %17 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[0] + y[1] + y[2] + y[3] + y[4] = 3
    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.constant 3 : index
        %5 = bmodelica.constant 4 : index
        %6 = bmodelica.load %0[%1] : !bmodelica.array<5x!bmodelica.real>
        %7 = bmodelica.load %0[%2] : !bmodelica.array<5x!bmodelica.real>
        %8 = bmodelica.load %0[%3] : !bmodelica.array<5x!bmodelica.real>
        %9 = bmodelica.load %0[%4] : !bmodelica.array<5x!bmodelica.real>
        %10 = bmodelica.load %0[%5] : !bmodelica.array<5x!bmodelica.real>
        %11 = bmodelica.add %6, %7 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %12 = bmodelica.add %11, %8 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %13 = bmodelica.add %12, %9 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %14 = bmodelica.add %13, %10 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %15 = bmodelica.constant #bmodelica.real<3.0>
        %16 = bmodelica.equation_side %14 : tuple<!bmodelica.real>
        %17 = bmodelica.equation_side %15 : tuple<!bmodelica.real>
        bmodelica.equation_sides %16, %17 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // x[0] - x[1] + x[2] + x[3] + x[4] = 2
    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.constant 3 : index
        %5 = bmodelica.constant 4 : index
        %6 = bmodelica.load %0[%1] : !bmodelica.array<5x!bmodelica.real>
        %7 = bmodelica.load %0[%2] : !bmodelica.array<5x!bmodelica.real>
        %8 = bmodelica.load %0[%3] : !bmodelica.array<5x!bmodelica.real>
        %9 = bmodelica.load %0[%4] : !bmodelica.array<5x!bmodelica.real>
        %10 = bmodelica.load %0[%5] : !bmodelica.array<5x!bmodelica.real>
        %11 = bmodelica.add %6, %7 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %12 = bmodelica.sub %11, %8 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %13 = bmodelica.add %12, %9 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %14 = bmodelica.add %13, %10 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %15 = bmodelica.constant #bmodelica.real<2.0>
        %16 = bmodelica.equation_side %14 : tuple<!bmodelica.real>
        %17 = bmodelica.equation_side %15 : tuple<!bmodelica.real>
        bmodelica.equation_sides %16, %17 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // y[0] + y[1] - y[2] + y[3] + y[4] = 3
    // CHECK-DAG: %[[t4:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t4"}
    %t4 = bmodelica.equation_template inductions = [] attributes {id = "t4"} {
        %0 = bmodelica.variable_get @y : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.constant 3 : index
        %5 = bmodelica.constant 4 : index
        %6 = bmodelica.load %0[%1] : !bmodelica.array<5x!bmodelica.real>
        %7 = bmodelica.load %0[%2] : !bmodelica.array<5x!bmodelica.real>
        %8 = bmodelica.load %0[%3] : !bmodelica.array<5x!bmodelica.real>
        %9 = bmodelica.load %0[%4] : !bmodelica.array<5x!bmodelica.real>
        %10 = bmodelica.load %0[%5] : !bmodelica.array<5x!bmodelica.real>
        %11 = bmodelica.add %6, %7 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %12 = bmodelica.add %11, %8 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %13 = bmodelica.sub %12, %9 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %14 = bmodelica.add %13, %10 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %15 = bmodelica.constant #bmodelica.real<3.0>
        %16 = bmodelica.equation_side %14 : tuple<!bmodelica.real>
        %17 = bmodelica.equation_side %15 : tuple<!bmodelica.real>
        bmodelica.equation_sides %16, %17 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // x[0] + x[1] + x[2] - x[3] + x[4] = 2
    // CHECK-DAG: %[[t5:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t5"}
    %t5 = bmodelica.equation_template inductions = [] attributes {id = "t5"} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.constant 3 : index
        %5 = bmodelica.constant 4 : index
        %6 = bmodelica.load %0[%1] : !bmodelica.array<5x!bmodelica.real>
        %7 = bmodelica.load %0[%2] : !bmodelica.array<5x!bmodelica.real>
        %8 = bmodelica.load %0[%3] : !bmodelica.array<5x!bmodelica.real>
        %9 = bmodelica.load %0[%4] : !bmodelica.array<5x!bmodelica.real>
        %10 = bmodelica.load %0[%5] : !bmodelica.array<5x!bmodelica.real>
        %11 = bmodelica.add %6, %7 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %12 = bmodelica.add %11, %8 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %13 = bmodelica.sub %12, %9 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %14 = bmodelica.add %13, %10 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %15 = bmodelica.constant #bmodelica.real<2.0>
        %16 = bmodelica.equation_side %14 : tuple<!bmodelica.real>
        %17 = bmodelica.equation_side %15 : tuple<!bmodelica.real>
        bmodelica.equation_sides %16, %17 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.main_model {
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]]
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]]
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]]
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t3]]
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t4]]
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t5]]
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [0,4]>} : !bmodelica.equation
        bmodelica.equation_instance %t1 : !bmodelica.equation
        bmodelica.equation_instance %t2 : !bmodelica.equation
        bmodelica.equation_instance %t3 : !bmodelica.equation
        bmodelica.equation_instance %t4: !bmodelica.equation
        bmodelica.equation_instance %t5 : !bmodelica.equation
    }
}
