// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// COM: l + fl = 0
// COM: fl = 0
// COM: h + fh = 0
// COM: fh = 0
// COM: for i in 0:4
// COM:   fl + f[i] + x[i] = 0
// COM: for i in 0:4
// COM:   fh + f[i] + y[i] = 0
// COM: for i in 0:4
// COM:   f[i] = 0

bmodelica.model @Test {
    bmodelica.variable @l : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @h : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @fl : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @fh : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @f : !bmodelica.variable<5x!bmodelica.real>

    // COM: l + fl = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @l : !bmodelica.real
        %1 = bmodelica.variable_get @fl : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: fl = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @fl : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: h + fh = 0
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @h : !bmodelica.real
        %1 = bmodelica.variable_get @fh : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    // COM: fh = 0
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @fh : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}

    // COM: fl + f[i] + x[i] = 0
    %t4 = bmodelica.equation_template inductions = [%i0] attributes {id = "t4"} {
        %0 = bmodelica.variable_get @fl : !bmodelica.real
        %1 = bmodelica.variable_get @f : tensor<5x!bmodelica.real>
        %2 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0] : tensor<5x!bmodelica.real>
        %4 = bmodelica.tensor_extract %2[%i0] : tensor<5x!bmodelica.real>
        %5 = bmodelica.add %0, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.add %5, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %7 = bmodelica.constant #bmodelica<real 0.0>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        %9 = bmodelica.equation_side %7 : tuple<!bmodelica.real>
        bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t4:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t4"}

    // COM: fh + f[i] + y[i] = 0
    %t5 = bmodelica.equation_template inductions = [%i0] attributes {id = "t5"} {
        %0 = bmodelica.variable_get @fh : !bmodelica.real
        %1 = bmodelica.variable_get @f : tensor<5x!bmodelica.real>
        %2 = bmodelica.variable_get @y : tensor<5x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0] : tensor<5x!bmodelica.real>
        %4 = bmodelica.tensor_extract %2[%i0] : tensor<5x!bmodelica.real>
        %5 = bmodelica.add %0, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.add %5, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %7 = bmodelica.constant #bmodelica<real 0.0>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        %9 = bmodelica.equation_side %7 : tuple<!bmodelica.real>
        bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t5:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t5"}

    // COM: f[i] = 0
    %t6 = bmodelica.equation_template inductions = [%i0] attributes {id = "t6"} {
        %0 = bmodelica.variable_get @f : tensor<5x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t6:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t6"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
        bmodelica.equation_instance %t1
        bmodelica.equation_instance %t2
        bmodelica.equation_instance %t3
        bmodelica.equation_instance %t4, indices = {[0,4]}
        bmodelica.equation_instance %t5, indices = {[0,4]}
        bmodelica.equation_instance %t6, indices = {[0,4]}

        // CHECK-DAG: bmodelica.equation_instance %[[t0]], match = @l
        // CHECK-DAG: bmodelica.equation_instance %[[t1]], match = @fl
        // CHECK-DAG: bmodelica.equation_instance %[[t2]], match = @h
        // CHECK-DAG: bmodelica.equation_instance %[[t3]], match = @fh
        // CHECK-DAG: bmodelica.equation_instance %[[t4]], indices = {[0,4]}, match = <@x, {[0,4]}>
        // CHECK-DAG: bmodelica.equation_instance %[[t5]], indices = {[0,4]}, match = <@y, {[0,4]}>
        // CHECK-DAG: bmodelica.equation_instance %[[t6]], indices = {[0,4]}, match = <@f, {[0,4]}>
    }
}
