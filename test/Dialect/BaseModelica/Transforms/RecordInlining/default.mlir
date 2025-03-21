// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @defaultOfRecord

bmodelica.function @defaultOfRecord {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.default @r {
        %0 = bmodelica.constant #bmodelica<real 0.0>
        %1 = bmodelica.constant #bmodelica<real 1.0>
        %2 = bmodelica.record_create %0, %1 : !bmodelica.real, !bmodelica.real -> !bmodelica<record @R>
        bmodelica.yield %2 : !bmodelica<record @R>
    }

    // CHECK:       bmodelica.default @r.x {
    // CHECK-NEXT:  %[[x:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-NEXT:  bmodelica.yield %[[x]]

    // CHECK:       bmodelica.default @r.y {
    // CHECK-NEXT:  %[[y:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK-NEXT:  bmodelica.yield %[[y]]
}
