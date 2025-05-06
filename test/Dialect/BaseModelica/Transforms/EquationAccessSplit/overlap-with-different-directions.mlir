// RUN: modelica-opt %s --split-input-file --split-overlapping-accesses --canonicalize | FileCheck %s

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<20x!bmodelica.real>

    // x[i + 5] = x[14 - i]
    %t0 = bmodelica.equation_template inductions = [%i] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<20x!bmodelica.real>
        %1 = bmodelica.constant 5 : index
        %2 = bmodelica.add %i, %1 : (index, index) -> index
        %3 = bmodelica.tensor_extract %0[%2] : tensor<20x!bmodelica.real>
        %4 = bmodelica.constant 14 : index
        %5 = bmodelica.sub %4, %i : (index, index) -> index
        %6 = bmodelica.tensor_extract %0[%5] : tensor<20x!bmodelica.real>
        %7 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[4,7]}, match = <@x, {[7,12]}>
    }

    // CHECK:       bmodelica.dynamic
    // CHECK-DAG:   bmodelica.equation_instance %[[t0]], indices = {[4,5]}, match = <@x, {[9,10]}>
    // CHECK-DAG:   bmodelica.equation_instance %[[t0]], indices = {[6,7]}, match = <@x, {[7,8],[11,12]}>
}
