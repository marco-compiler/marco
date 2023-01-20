// RUN: modelica-opt %s --canonicalize | FileCheck %s

// Check that constants get materialized inside the parent equation.

// CHECK-LABEL: @Test
// CHECK:       modelica.algorithm {
// CHECK-NEXT:      %[[cst:.*]] = modelica.constant #modelica.real<3.000000e+00>
// CHECK-NEXT:      modelica.store %{{.*}}[], %[[cst]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real>
    modelica.yield
} body {
^bb0(%arg0: !modelica.array<!modelica.real>):
    modelica.algorithm {
        %0 = modelica.constant #modelica.real<1.0>
        %1 = modelica.constant #modelica.real<2.0>
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.store %arg0[], %2 : !modelica.array<!modelica.real>
    }
}
