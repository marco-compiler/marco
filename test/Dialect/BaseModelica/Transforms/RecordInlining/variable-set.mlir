// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// CHECK-LABEL: @Test

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Test {
    bmodelica.variable @r1 : !bmodelica.variable<!bmodelica<record @R>>
    bmodelica.variable @r2 : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @r1 : !bmodelica<record @R>
        bmodelica.variable_set @r2, %0 : !bmodelica<record @R>
    }

    // CHECK:       bmodelica.algorithm
    // CHECK-DAG:   %[[r1_x:.*]] = bmodelica.variable_get @r1.x
    // CHECK-DAG:   bmodelica.variable_set @r2.x, %[[r1_x]]
    // CHECK-DAG:   %[[r1_y:.*]] = bmodelica.variable_get @r1.y
    // CHECK-DAG:   bmodelica.variable_set @r2.y, %[[r1_y]]
}
