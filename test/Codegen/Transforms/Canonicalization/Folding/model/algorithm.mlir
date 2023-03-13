// RUN: modelica-opt %s --canonicalize | FileCheck %s

// Check that constants get materialized inside the parent equation.

// CHECK-LABEL: @Test
// CHECK:       modelica.algorithm {
// CHECK-NEXT:      %[[cst:.*]] = modelica.constant #modelica.real<3.000000e+00>
// CHECK-NEXT:      modelica.variable_set @x, %[[cst]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.real>

    modelica.algorithm {
        %0 = modelica.constant #modelica.real<1.0>
        %1 = modelica.constant #modelica.real<2.0>
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @x, %2 : !modelica.real
    }
}
