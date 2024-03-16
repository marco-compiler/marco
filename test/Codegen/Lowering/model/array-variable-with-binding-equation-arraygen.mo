// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       modelica.binding_equation @x {
// CHECK-DAG:       %[[cst0:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG:       %[[cst1:.*]] = modelica.constant #modelica.int<2>
// CHECK-DAG:       %[[cst2:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG:       %[[cst3:.*]] = modelica.constant #modelica.int<2>
// CHECK-DAG:       %[[cst4:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG:       %[[cst5:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_from_elements %[[cst0]], %[[cst1]], %[[cst2]], %[[cst3]], %[[cst4]], %[[cst5]] : !modelica.int, !modelica.int, !modelica.int, !modelica.int, !modelica.int, !modelica.int -> <3x2x!modelica.real>
// CHECK-NEXT:      modelica.yield %[[array]]
// CHECK-NEXT:  }

model Test
	Real[3,2] x = {{1,2} for i in 1:3};
equation
end Test;
