// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       bmodelica.binding_equation @x {
// CHECK-DAG:       %[[cst0:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-DAG:       %[[cst1:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-DAG:       %[[cst2:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-DAG:       %[[cst3:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-DAG:       %[[cst4:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-DAG:       %[[cst5:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-NEXT:      %[[array:.*]] = bmodelica.array_from_elements %[[cst0]], %[[cst1]], %[[cst2]], %[[cst3]], %[[cst4]], %[[cst5]] : !bmodelica.int, !bmodelica.int, !bmodelica.int, !bmodelica.int, !bmodelica.int, !bmodelica.int -> <3x2x!bmodelica.real>
// CHECK-NEXT:      bmodelica.yield %[[array]]
// CHECK-NEXT:  }

model Test
	Real[3,2] x = {{1,2} for i in 1:3};
equation
end Test;
