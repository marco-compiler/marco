// RUN: modelica-opt %s --split-input-file --split-overlapping-accesses --canonicalize | FileCheck %s

// CHECK-LABEL: @Backward

bmodelica.model @Backward {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>

    // x[i] = x[i - 1]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.real>
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.sub %i0, %2 : (index, index) -> index
        %4 = bmodelica.tensor_extract %0[%3] : tensor<10x!bmodelica.real>
        %5 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // x[0] = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<10x!bmodelica.real>
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[1,9]}, match = <@x, {[1,9]}>
        bmodelica.equation_instance %t1, match = <@x, {[0,0]}>
    }

    // CHECK:       bmodelica.dynamic
    // CHECK-DAG:   bmodelica.equation_instance %[[t0]], indices = {[1,9]}, match = <@x, {[1,9]}>
    // CHECK-DAG:   bmodelica.equation_instance %[[t1]], match = <@x, {[0,0]}>
}

// -----

// CHECK-LABEL: @Forward

bmodelica.model @Forward {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>

    // x[i] = x[i + 1]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.real>
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.add %i0, %2 : (index, index) -> index
        %4 = bmodelica.tensor_extract %0[%3] : tensor<10x!bmodelica.real>
        %5 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // x[9] = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.constant 9 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<10x!bmodelica.real>
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[0,8]}, match = <@x, {[0,8]}>
        bmodelica.equation_instance %t1, match = <@x, {[9,9]}>
    }

    // CHECK:       bmodelica.dynamic
    // CHECK-DAG:   bmodelica.equation_instance %[[t0]], indices = {[0,8]}, match = <@x, {[0,8]}>
    // CHECK-DAG:   bmodelica.equation_instance %[[t1]], match = <@x, {[9,9]}>
}
