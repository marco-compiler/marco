// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       modelica.binding_equation @x {
// CHECK-DAG:       %[[cst0:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG:       %[[cst1:.*]] = modelica.constant #modelica.real<2.000000e+00>
// CHECK-DAG:       %[[cst2:.*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_from_elements %[[cst0]], %[[cst1]], %[[cst2]] : !modelica.int, !modelica.real, !modelica.int -> !modelica.array<3x!modelica.real>
// CHECK-NEXT:      modelica.yield %[[array]]
// CHECK-NEXT:  }

model Test
	Real[3] x = {1, 2.0, 3};
equation
end Test;
