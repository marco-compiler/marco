// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @Test

bmodelica.function @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @r : !bmodelica.variable<2x!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %2 = bmodelica.record_create %0 : !bmodelica.real -> !bmodelica<record @R>
        %3 = bmodelica.tensor_broadcast %2 : !bmodelica<record @R> -> tensor<2x!bmodelica<record @R>>
        bmodelica.variable_set @r, %3 : tensor<2x!bmodelica<record @R>>
    }

    // CHECK:       bmodelica.algorithm
    // CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.real
    // CHECK-DAG:   %[[broadcast:.*]] = bmodelica.tensor_broadcast %[[x]] : !bmodelica.real -> tensor<2x!bmodelica.real>
    // CHECK-DAG:   %[[unbounded:.*]] = bmodelica.unbounded_range
    // CHECK-DAG:   %[[r_x:.*]] = bmodelica.variable_get @r.x : tensor<2x!bmodelica.real>
    // CHECK:       %[[r_x_insert:.*]] = bmodelica.tensor_insert %[[broadcast]], %[[r_x]][%[[unbounded]]]
    // CHECK:       bmodelica.variable_set @r.x, %[[r_x_insert]]
}
