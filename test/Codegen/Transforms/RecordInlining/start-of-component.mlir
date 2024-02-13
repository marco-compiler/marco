// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// CHECK:       modelica.start @r.x {
// CHECK-NEXT:      %[[x_start:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK-NEXT:      modelica.yield %[[x_start]]
// CHECK-NEXT:  }

// CHECK:       modelica.start @r.y {
// CHECK-NEXT:      %[[y_start:.*]] = modelica.constant #modelica.real<2.000000e+00>
// CHECK-NEXT:      modelica.yield %[[y_start]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.model @M {
    modelica.variable @r : !modelica.variable<!modelica<record @R>>

    modelica.start @r::@x {
        %0 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
        modelica.yield %0 : !modelica.real
    } {each = false, fixed = false}

    modelica.start @r::@y {
        %0 = modelica.constant #modelica.real<2.000000e+00> : !modelica.real
        modelica.yield %0 : !modelica.real
    } {each = false, fixed = false}
}
