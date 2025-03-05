// RUN: modelica-opt %s --split-input-file --single-valued-induction-elimination | FileCheck %s

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x3x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0, %i1] {
        %0 = bmodelica.variable_get @x : tensor<2x3x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<2x3x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK:   %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i1:.*]]] {
    // CHECK:       %[[i0:.*]] = bmodelica.constant 0 : index
    // CHECK:       bmodelica.tensor_extract %{{.*}}[%[[i0]], %[[i1]]]

    // CHECK:   %[[t1:.*]] = bmodelica.equation_template inductions = [%[[i1:.*]]] {
    // CHECK:       %[[i0:.*]] = bmodelica.constant 1 : index
    // CHECK:       bmodelica.tensor_extract %{{.*}}[%[[i0]], %[[i1]]]

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[0,0][0,2]}
        bmodelica.equation_instance %t0, indices = {[1,1][0,2]}
    }

    // CHECK:     bmodelica.dynamic
    // CHECK-DAG: bmodelica.equation_instance %[[t0]], indices = {[0,2]}
    // CHECK-DAG: bmodelica.equation_instance %[[t1]], indices = {[0,2]}
}
