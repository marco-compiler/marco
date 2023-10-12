// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       modelica.binding_equation @x {
// CHECK-DAG:       %[[cst0:.*]] = modelica.constant #modelica.int<1234>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_broadcast %[[cst0]] : !modelica.int -> !modelica.array<3x!modelica.real>
// CHECK-NEXT:      modelica.yield %[[array]]
// CHECK-NEXT:  }

model Test
	Real[3] x = {1234 for i in 1:3};
equation
end Test;
