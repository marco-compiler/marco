// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @startOfRecord

bmodelica.model @startOfRecord {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.start @r {
        %0 = bmodelica.constant #bmodelica<real 0.0>
        %1 = bmodelica.constant #bmodelica<real 1.0>
        %2 = bmodelica.record_create %0, %1 : !bmodelica.real, !bmodelica.real -> !bmodelica<record @R>
        bmodelica.yield %2 : !bmodelica<record @R>
    } {each = false, fixed = true}

    // CHECK:       bmodelica.start @r.x
    // CHECK-NEXT:  %[[x:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-NEXT:  bmodelica.yield %[[x]]

    // CHECK:       bmodelica.start @r.y
    // CHECK-NEXT:  %[[y:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK-NEXT:  bmodelica.yield %[[y]]
}

// -----

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @startOfComponent

bmodelica.model @startOfComponent {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.start @r::@x {
        %0 = bmodelica.constant #bmodelica<real 1.000000e+00> : !bmodelica.real
        bmodelica.yield %0 : !bmodelica.real
    } {each = false, fixed = false}

    bmodelica.start @r::@y {
        %0 = bmodelica.constant #bmodelica<real 2.000000e+00> : !bmodelica.real
        bmodelica.yield %0 : !bmodelica.real
    } {each = false, fixed = false}

    // CHECK:       bmodelica.start @r.x
    // CHECK-NEXT:  %[[x_start:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK-NEXT:  bmodelica.yield %[[x_start]]

    // CHECK:       bmodelica.start @r.y
    // CHECK-NEXT:  %[[y_start:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
    // CHECK-NEXT:  bmodelica.yield %[[y_start]]
}
