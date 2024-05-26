// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       bmodelica.binding_equation @x {
// CHECK-DAG:       %[[cst0:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-DAG:       %[[cst1:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-DAG:       %[[cst2:.*]] = bmodelica.constant #bmodelica<int 3>
// CHECK-NEXT:      %[[tensor:.*]] = bmodelica.tensor_from_elements %[[cst0]], %[[cst1]], %[[cst2]] : !bmodelica.int, !bmodelica.real, !bmodelica.int -> tensor<3x!bmodelica.real>
// CHECK-NEXT:      bmodelica.yield %[[tensor]]
// CHECK-NEXT:  }

model Test
	Real[3] x = {1, 2.0, 3};
equation
end Test;
