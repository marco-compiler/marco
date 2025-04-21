// RUN: modelica-opt %s --split-input-file --derivatives-materialization | FileCheck %s

// CHECK-LABEL: @ScalarVariable

bmodelica.model @ScalarVariable der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.der %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.constant #bmodelica<real 0.000000e+00>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = []
    // CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[der_x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]

    // CHECK-NOT: bmodelica.equation_template

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
    }

    // CHECK:       bmodelica.dynamic
    // CHECK-NEXT:  bmodelica.equation_instance %[[t0]]
    // CHECK-NOT: bmodelica.equation_instance
}

// -----

// CHECK-LABEL: @Array1DNoOverlap

// CHECK:                         der = [<@x, @der_x, {[0,9]}
bmodelica.model @Array1DNoOverlap der = [<@x, @der_x, {[0,4]}>] {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<10x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0] {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]]
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[view:.*]] = bmodelica.tensor_view %[[der_x]][%[[i0]]]
    // CHECK-DAG:       %[[extract:.*]] = bmodelica.tensor_extract %[[view]][]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
    // CHECK-DAG:       bmodelica.equation_sides %[[lhs]], %{{.*}}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[5,9]}
    }

    // CHECK: bmodelica.equation_instance %[[t0]], indices = {[5,9]}
}

// -----

// CHECK-LABEL: @Array1DPartialOverlap

// CHECK:                              der = [<@x, @der_x, {[0,6]}
bmodelica.model @Array1DPartialOverlap der = [<@x, @der_x, {[0,4]}>] {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<10x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0] {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]]
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[view:.*]] = bmodelica.tensor_view %[[der_x]][%[[i0]]]
    // CHECK-DAG:       %[[extract:.*]] = bmodelica.tensor_extract %[[view]][]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
    // CHECK-DAG:       bmodelica.equation_sides %[[lhs]], %{{.*}}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[2,6]}
    }

    // CHECK: bmodelica.equation_instance %[[t0]], indices = {[2,6]}
}

// -----

// CHECK-LABEL: @Array2DPartialOverlap

// CHECK:                              der = [<@x, @der_x, {[0,4][0,6],[5,9][2,5]}
bmodelica.model @Array2DPartialOverlap der = [<@x, @der_x, {[0,4][0,6]}>] {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<10x20x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0, %i1] {
        %0 = bmodelica.variable_get @x : tensor<10x20x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<10x20x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]]
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[view:.*]] = bmodelica.tensor_view %[[der_x]][%[[i0]]]
    // CHECK-DAG:       %[[extract:.*]] = bmodelica.tensor_extract %[[view]][]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
    // CHECK-DAG:       bmodelica.equation_sides %[[lhs]], %{{.*}}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[3,9][2,5]}
    }

    // CHECK: bmodelica.equation_instance %[[t0]], indices = {[3,9][2,5]}
}