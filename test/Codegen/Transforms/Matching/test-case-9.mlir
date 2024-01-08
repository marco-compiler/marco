// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// for i in 0:4
//   x[i] - y[i] = 0
// x[0] + x[1] + x[2] + x[3] + x[4] = 2
// y[0] + y[1] + y[2] + y[3] + y[4] = 3
// x[0] - x[1] + x[2] + x[3] + x[4] = 2
// y[0] + y[1] - y[2] + y[3] + y[4] = 3
// x[0] + x[1] + x[2] - x[3] + x[4] = 2

modelica.model @Test {
    modelica.variable @x : !modelica.variable<5x!modelica.real>
    modelica.variable @y : !modelica.variable<5x!modelica.real>

    // x[i] - y[i] = 0
    // CHECK-DAG: %[[t0:.*]] = modelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}
    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.variable_get @y : !modelica.array<5x!modelica.real>
        %2 = modelica.load %0[%i0] : !modelica.array<5x!modelica.real>
        %3 = modelica.load %1[%i0] : !modelica.array<5x!modelica.real>
        %4 = modelica.sub %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.constant #modelica.real<0.0>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // x[0] + x[1] + x[2] + x[3] + x[4] = 2
    // CHECK-DAG: %[[t1:.*]] = modelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = modelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.constant 1 : index
        %3 = modelica.constant 2 : index
        %4 = modelica.constant 3 : index
        %5 = modelica.constant 4 : index
        %6 = modelica.load %0[%1] : !modelica.array<5x!modelica.real>
        %7 = modelica.load %0[%2] : !modelica.array<5x!modelica.real>
        %8 = modelica.load %0[%3] : !modelica.array<5x!modelica.real>
        %9 = modelica.load %0[%4] : !modelica.array<5x!modelica.real>
        %10 = modelica.load %0[%5] : !modelica.array<5x!modelica.real>
        %11 = modelica.add %6, %7 : (!modelica.real, !modelica.real) -> !modelica.real
        %12 = modelica.add %11, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %13 = modelica.add %12, %9 : (!modelica.real, !modelica.real) -> !modelica.real
        %14 = modelica.add %13, %10 : (!modelica.real, !modelica.real) -> !modelica.real
        %15 = modelica.constant #modelica.real<2.0>
        %16 = modelica.equation_side %14 : tuple<!modelica.real>
        %17 = modelica.equation_side %15 : tuple<!modelica.real>
        modelica.equation_sides %16, %17 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // y[0] + y[1] + y[2] + y[3] + y[4] = 3
    // CHECK-DAG: %[[t2:.*]] = modelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = modelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = modelica.variable_get @y : !modelica.array<5x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.constant 1 : index
        %3 = modelica.constant 2 : index
        %4 = modelica.constant 3 : index
        %5 = modelica.constant 4 : index
        %6 = modelica.load %0[%1] : !modelica.array<5x!modelica.real>
        %7 = modelica.load %0[%2] : !modelica.array<5x!modelica.real>
        %8 = modelica.load %0[%3] : !modelica.array<5x!modelica.real>
        %9 = modelica.load %0[%4] : !modelica.array<5x!modelica.real>
        %10 = modelica.load %0[%5] : !modelica.array<5x!modelica.real>
        %11 = modelica.add %6, %7 : (!modelica.real, !modelica.real) -> !modelica.real
        %12 = modelica.add %11, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %13 = modelica.add %12, %9 : (!modelica.real, !modelica.real) -> !modelica.real
        %14 = modelica.add %13, %10 : (!modelica.real, !modelica.real) -> !modelica.real
        %15 = modelica.constant #modelica.real<3.0>
        %16 = modelica.equation_side %14 : tuple<!modelica.real>
        %17 = modelica.equation_side %15 : tuple<!modelica.real>
        modelica.equation_sides %16, %17 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // x[0] - x[1] + x[2] + x[3] + x[4] = 2
    // CHECK-DAG: %[[t3:.*]] = modelica.equation_template inductions = [] attributes {id = "t3"}
    %t3 = modelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.constant 1 : index
        %3 = modelica.constant 2 : index
        %4 = modelica.constant 3 : index
        %5 = modelica.constant 4 : index
        %6 = modelica.load %0[%1] : !modelica.array<5x!modelica.real>
        %7 = modelica.load %0[%2] : !modelica.array<5x!modelica.real>
        %8 = modelica.load %0[%3] : !modelica.array<5x!modelica.real>
        %9 = modelica.load %0[%4] : !modelica.array<5x!modelica.real>
        %10 = modelica.load %0[%5] : !modelica.array<5x!modelica.real>
        %11 = modelica.add %6, %7 : (!modelica.real, !modelica.real) -> !modelica.real
        %12 = modelica.sub %11, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %13 = modelica.add %12, %9 : (!modelica.real, !modelica.real) -> !modelica.real
        %14 = modelica.add %13, %10 : (!modelica.real, !modelica.real) -> !modelica.real
        %15 = modelica.constant #modelica.real<2.0>
        %16 = modelica.equation_side %14 : tuple<!modelica.real>
        %17 = modelica.equation_side %15 : tuple<!modelica.real>
        modelica.equation_sides %16, %17 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // y[0] + y[1] - y[2] + y[3] + y[4] = 3
    // CHECK-DAG: %[[t4:.*]] = modelica.equation_template inductions = [] attributes {id = "t4"}
    %t4 = modelica.equation_template inductions = [] attributes {id = "t4"} {
        %0 = modelica.variable_get @y : !modelica.array<5x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.constant 1 : index
        %3 = modelica.constant 2 : index
        %4 = modelica.constant 3 : index
        %5 = modelica.constant 4 : index
        %6 = modelica.load %0[%1] : !modelica.array<5x!modelica.real>
        %7 = modelica.load %0[%2] : !modelica.array<5x!modelica.real>
        %8 = modelica.load %0[%3] : !modelica.array<5x!modelica.real>
        %9 = modelica.load %0[%4] : !modelica.array<5x!modelica.real>
        %10 = modelica.load %0[%5] : !modelica.array<5x!modelica.real>
        %11 = modelica.add %6, %7 : (!modelica.real, !modelica.real) -> !modelica.real
        %12 = modelica.add %11, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %13 = modelica.sub %12, %9 : (!modelica.real, !modelica.real) -> !modelica.real
        %14 = modelica.add %13, %10 : (!modelica.real, !modelica.real) -> !modelica.real
        %15 = modelica.constant #modelica.real<3.0>
        %16 = modelica.equation_side %14 : tuple<!modelica.real>
        %17 = modelica.equation_side %15 : tuple<!modelica.real>
        modelica.equation_sides %16, %17 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    // x[0] + x[1] + x[2] - x[3] + x[4] = 2
    // CHECK-DAG: %[[t5:.*]] = modelica.equation_template inductions = [] attributes {id = "t5"}
    %t5 = modelica.equation_template inductions = [] attributes {id = "t5"} {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.constant 1 : index
        %3 = modelica.constant 2 : index
        %4 = modelica.constant 3 : index
        %5 = modelica.constant 4 : index
        %6 = modelica.load %0[%1] : !modelica.array<5x!modelica.real>
        %7 = modelica.load %0[%2] : !modelica.array<5x!modelica.real>
        %8 = modelica.load %0[%3] : !modelica.array<5x!modelica.real>
        %9 = modelica.load %0[%4] : !modelica.array<5x!modelica.real>
        %10 = modelica.load %0[%5] : !modelica.array<5x!modelica.real>
        %11 = modelica.add %6, %7 : (!modelica.real, !modelica.real) -> !modelica.real
        %12 = modelica.add %11, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %13 = modelica.sub %12, %9 : (!modelica.real, !modelica.real) -> !modelica.real
        %14 = modelica.add %13, %10 : (!modelica.real, !modelica.real) -> !modelica.real
        %15 = modelica.constant #modelica.real<2.0>
        %16 = modelica.equation_side %14 : tuple<!modelica.real>
        %17 = modelica.equation_side %15 : tuple<!modelica.real>
        modelica.equation_sides %16, %17 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        // CHECK-DAG: modelica.matched_equation_instance %[[t0]]
        // CHECK-DAG: modelica.matched_equation_instance %[[t1]]
        // CHECK-DAG: modelica.matched_equation_instance %[[t2]]
        // CHECK-DAG: modelica.matched_equation_instance %[[t3]]
        // CHECK-DAG: modelica.matched_equation_instance %[[t4]]
        // CHECK-DAG: modelica.matched_equation_instance %[[t5]]
        modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,4]>} : !modelica.equation
        modelica.equation_instance %t1 : !modelica.equation
        modelica.equation_instance %t2 : !modelica.equation
        modelica.equation_instance %t3 : !modelica.equation
        modelica.equation_instance %t4: !modelica.equation
        modelica.equation_instance %t5 : !modelica.equation
    }
}
