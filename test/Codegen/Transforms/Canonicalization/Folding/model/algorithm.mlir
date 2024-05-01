// RUN: modelica-opt %s --canonicalize | FileCheck %s

// Check that constants get materialized inside the parent equation.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.algorithm {
// CHECK-NEXT:      %[[cst:.*]] = bmodelica.constant #bmodelica.real<3.000000e+00>
// CHECK-NEXT:      bmodelica.variable_set @x, %[[cst]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    bmodelica.algorithm {
        %0 = bmodelica.constant #bmodelica.real<1.0>
        %1 = bmodelica.constant #bmodelica.real<2.0>
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @x, %2 : !bmodelica.real
    }
}
