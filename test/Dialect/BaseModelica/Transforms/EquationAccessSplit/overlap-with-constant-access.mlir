// RUN: modelica-opt %s --split-input-file --split-overlapping-accesses --canonicalize | FileCheck %s

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<7x8x!bmodelica.real>

    // x[i0,i1] = 2 * x[2,2] - 4
    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<7x8x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<7x8x!bmodelica.real>
        %2 = bmodelica.constant 2 : index
        %3 = bmodelica.tensor_extract %0[%2, %2] : tensor<7x8x!bmodelica.real>
        %4 = bmodelica.constant #bmodelica<int 2>
        %5 = bmodelica.constant #bmodelica<int 4>
        %6 = bmodelica.mul %4, %3 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
        %7 = bmodelica.sub %6, %5 : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
        %8 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %9 = bmodelica.equation_side %7 : tuple<!bmodelica.real>
        bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}, %{{.*}}] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[0,6][0,7]}, match = <@x, {[0,6][0,7]}>
    }

    // CHECK:       bmodelica.dynamic
    // CHECK-DAG:   bmodelica.equation_instance %[[t0]], indices = {[0,1][0,7],[2,2][0,1],[2,2][3,7],[3,6][0,7]}, match = <@x, {[0,1][0,7],[2,2][0,1],[2,2][3,7],[3,6][0,7]}>
    // CHECK-DAG:   bmodelica.equation_instance %[[t0]], indices = {[2,2][2,2]}, match = <@x, {[2,2][2,2]}>
}
