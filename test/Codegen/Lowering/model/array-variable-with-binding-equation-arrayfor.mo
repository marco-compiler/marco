// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       bmodelica.binding_equation @x {
// CHECK-DAG:       %[[cst0:.*]] = bmodelica.constant #bmodelica<int 1234>
// CHECK-NEXT:      %[[tensor:.*]] = bmodelica.tensor_broadcast %[[cst0]] : !bmodelica.int -> tensor<3x!bmodelica.real>
// CHECK-NEXT:      bmodelica.yield %[[tensor]]
// CHECK-NEXT:  }

model Test
	Real[3] x = {1234 for i in 1:3};
equation
end Test;
