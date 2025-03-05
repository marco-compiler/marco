// RUN: modelica-opt %s --split-input-file --derivatives-materialization | FileCheck %s

// CHECK-LABEL: @arrayVariable

bmodelica.model @arrayVariable {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x20x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<10x20x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 3.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"}
    // CHECK:           %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK:           %[[view:.*]] = bmodelica.tensor_view %[[der_x]][%[[i0]], %[[i1]]]
    // CHECK:           %[[extract:.*]] = bmodelica.tensor_extract %[[view]][]
    // CHECK:           %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %{{.*}}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[3,5][12,14]}
    }

    // CHECK: bmodelica.equation_instance %[[t0]], indices = {[3,5][12,14]}
}

// -----

// COM: Array variable with not all indices being derived.

// CHECK-LABEL: @partialArrayVariable

bmodelica.model @partialArrayVariable {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x20x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<10x20x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 3.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] {
    // CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[extract:.*]] = bmodelica.tensor_extract %[[der_x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[3,5][12,14]}
    }

    // CHECK: bmodelica.dynamic
    // CHECK-DAG:  bmodelica.equation_instance %[[t0]], indices = {[0,2][0,19],[3,5][0,11],[3,5][15,19],[6,9][0,19]}
}
