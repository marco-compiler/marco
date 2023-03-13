// RUN: modelica-opt %s --canonicalize | FileCheck %s

// Check that constants get materialized inside the parent equation.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[cst:.*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[cst]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[cst]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.equation {
        %0 = modelica.constant #modelica.real<1.0>
        %1 = modelica.constant #modelica.real<4.0>
        %2 = modelica.constant #modelica.real<2.0>
        %3 = modelica.constant #modelica.real<3.0>
        %4 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.add %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }
}
