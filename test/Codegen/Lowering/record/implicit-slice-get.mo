// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK-DAG:           %[[r:.*]] = bmodelica.variable_get @r : tensor<3x!bmodelica<record @R>>
// CHECK-DAG:           %[[r_x:.*]] = bmodelica.component_get %[[r]], @x : tensor<3x!bmodelica<record @R>> -> tensor<3x2x!bmodelica.real>
// CHECK-DAG:           %[[unbounded_range:.*]] = bmodelica.unbounded_range
// CHECK-DAG:           %[[one:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-DAG:           %[[minus_one:.*]] = bmodelica.constant -1 : index
// CHECK-DAG:           %[[subscript:.*]] = bmodelica.add %[[one]], %[[minus_one]]
// CHECK-NEXT:          %[[view:.*]] = bmodelica.tensor_view %[[r_x]][%[[unbounded_range]], %[[subscript]]]
// CHECK-NEXT:          bmodelica.variable_set @x, %[[view]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real[2] x;
end R;

function Foo
    input R[3] r;
    output Real[3] x;
algorithm
    x := r.x[1];
end Foo;
