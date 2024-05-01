// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// CHECK:       bmodelica.start @r.x {
// CHECK-NEXT:      %[[x_start:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK-NEXT:      bmodelica.yield %[[x_start]]
// CHECK-NEXT:  }

// CHECK:       bmodelica.start @r.y {
// CHECK-NEXT:      %[[y_start:.*]] = bmodelica.constant #bmodelica.real<2.000000e+00>
// CHECK-NEXT:      bmodelica.yield %[[y_start]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.model @M {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.start @r::@x {
        %0 = bmodelica.constant #bmodelica.real<1.000000e+00> : !bmodelica.real
        bmodelica.yield %0 : !bmodelica.real
    } {each = false, fixed = false}

    bmodelica.start @r::@y {
        %0 = bmodelica.constant #bmodelica.real<2.000000e+00> : !bmodelica.real
        bmodelica.yield %0 : !bmodelica.real
    } {each = false, fixed = false}
}
