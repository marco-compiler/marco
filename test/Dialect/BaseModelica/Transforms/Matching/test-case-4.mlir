// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// COM: l + f[0] = 0
// COM: f[0] = 0
// COM: for i in 0:4
// COM:   x[i] + f[i] + f[i + 1] = 0
// COM: for i in 1:4
// COM:   f[i] = 0
// COM: h + f[5] = 0
// COM: f[5] = 0

bmodelica.model @Test {
    bmodelica.variable @l : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @h : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @f : !bmodelica.variable<6x!bmodelica.real>

    // COM: l + f[0] = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @l : !bmodelica.real
        %1 = bmodelica.variable_get @f : tensor<6x!bmodelica.real>
        %2 = bmodelica.constant 0 : index
        %3 = bmodelica.tensor_extract %1[%2] : tensor<6x!bmodelica.real>
        %4 = bmodelica.add %0, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %5 = bmodelica.constant #bmodelica<real 0.0>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: f[0] = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @f : tensor<6x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<6x!bmodelica.real>
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: x[i] + f[i] + f[i + 1] = 0
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %1 = bmodelica.variable_get @f : tensor<6x!bmodelica.real>
        %2 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.real>
        %3 = bmodelica.tensor_extract %1[%i0] : tensor<6x!bmodelica.real>
        %4 = bmodelica.constant 1 : index
        %5 = bmodelica.add %i0, %4 : (index, index) -> index
        %6 = bmodelica.tensor_extract %1[%5] : tensor<6x!bmodelica.real>
        %7 = bmodelica.add %2, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %8 = bmodelica.add %7, %6 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %9 = bmodelica.constant #bmodelica<real 0.0>
        %10 = bmodelica.equation_side %8 : tuple<!bmodelica.real>
        %11 = bmodelica.equation_side %9 : tuple<!bmodelica.real>
        bmodelica.equation_sides %10, %11 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}

    // COM: f[i] = 0
    %t3 = bmodelica.equation_template inductions = [%i0] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @f : tensor<6x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<6x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t3"}

    // COM: h + f[5] = 0
    %t4 = bmodelica.equation_template inductions = [] attributes {id = "t4"} {
        %0 = bmodelica.variable_get @h : !bmodelica.real
        %1 = bmodelica.variable_get @f : tensor<6x!bmodelica.real>
        %2 = bmodelica.constant 5 : index
        %3 = bmodelica.tensor_extract %1[%2] : tensor<6x!bmodelica.real>
        %4 = bmodelica.add %0, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %5 = bmodelica.constant #bmodelica<real 0.0>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t4:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t4"}

    // COM: f[5] = 0
    %t5 = bmodelica.equation_template inductions = [] attributes {id = "t5"} {
        %0 = bmodelica.variable_get @f : tensor<6x!bmodelica.real>
        %1 = bmodelica.constant 5 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<6x!bmodelica.real>
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t5:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t5"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
        bmodelica.equation_instance %t1
        bmodelica.equation_instance %t2, indices = {[0,4]}
        bmodelica.equation_instance %t3, indices = {[1,4]}
        bmodelica.equation_instance %t4
        bmodelica.equation_instance %t5

        // CHECK-DAG: bmodelica.equation_instance %[[t0]], match = @l
        // CHECK-DAG: bmodelica.equation_instance %[[t1]], match = <@f, {[0,0]}>
        // CHECK-DAG: bmodelica.equation_instance %[[t2]], indices = {[0,4]}, match = <@x, {[0,4]}>
        // CHECK-DAG: bmodelica.equation_instance %[[t3]], indices = {[1,4]}, match = <@f, {[1,4]}>
        // CHECK-DAG: bmodelica.equation_instance %[[t4]], match = @h
        // CHECK-DAG: bmodelica.equation_instance %[[t5]], match = <@f, {[5,5]}>
    }
}
