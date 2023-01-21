// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[x:.*]]: !modelica.array<3x!modelica.real>):
// CHECK-NEXT:    modelica.binding_equation (%[[x]] : !modelica.array<3x!modelica.real>) {
// CHECK-NEXT:      %[[cst0:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[cst1:.*]] = modelica.constant #modelica.real<2.000000e+00>
// CHECK-NEXT:      %[[cst2:.*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_from_elements %[[cst0]], %[[cst1]], %[[cst2]] : !modelica.int, !modelica.real, !modelica.int -> !modelica.array<3x!modelica.real>
// CHECK-NEXT:      modelica.yield %[[array]]
// CHECK-NEXT:    }

model Test
	Real[3] x = {1, 2.0, 3};
equation
end Test;
