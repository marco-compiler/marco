// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Foo
// CHECK:       bmodelica.algorithm {
// CHECK-DAG:       %[[one:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-DAG:       %[[minus_one:.*]] = bmodelica.constant -1 : index
// CHECK-DAG:       %[[subscript:.*]] = bmodelica.add %[[one]], %[[minus_one]]
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x : tensor<2x3x4x!bmodelica.real>
// CHECK:           %[[view:.*]] = bmodelica.tensor_view %[[x]][%[[subscript]]]
// CHECK-NEXT:      bmodelica.variable_set @y, %[[view]]
// CHECK-NEXT:  }

function Foo
    input Real[2,3,4] x;
    output Real[3,4] y;
algorithm
    y := x[1];
end Foo;
