// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<!modelica.real>
// CHECK:       modelica.start @r.x {
// CHECK-NEXT:      %[[x:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      modelica.yield %[[x]]
// CHECK-NEXT:  }
// CHECK:       modelica.start @r.y {
// CHECK-NEXT:      %[[y:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK-NEXT:      modelica.yield %[[y]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.model @Test {
    modelica.variable @r : !modelica.variable<!modelica.record<@R>>

    modelica.start @r {
        %0 = modelica.constant #modelica.real<0.0>
        %1 = modelica.constant #modelica.real<1.0>
        %2 = modelica.record_create %0, %1 : !modelica.real, !modelica.real -> !modelica.record<@R>
        modelica.yield %2 : !modelica.record<@R>
    } {each = false, fixed = true}
}
