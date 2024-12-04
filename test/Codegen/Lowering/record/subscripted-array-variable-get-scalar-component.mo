// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK:               %[[one:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK:               %[[minus_one:.*]] = bmodelica.constant -1 : index
// CHECK:               %[[subscript:.*]] = bmodelica.add %[[one]], %[[minus_one]]
// CHECK:               %[[r:.*]] = bmodelica.variable_get @r : tensor<3x!bmodelica<record @R>>
// CHECK:               %[[view:.*]] = bmodelica.tensor_view %[[r]][%[[subscript]]] : tensor<3x!bmodelica<record @R>>, index -> tensor<!bmodelica<record @R>>
// CHECK:               %[[extract:.*]] = bmodelica.tensor_extract %[[view]][] : tensor<!bmodelica<record @R>>
// CHECK:               %[[component:.*]] = bmodelica.component_get %[[extract]], @x : !bmodelica<record @R> -> !bmodelica.real
// CHECK:               bmodelica.variable_set @x, %[[component]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real x;
end R;

function Foo
    input R[3] r;
    output Real x;
algorithm
    x := r[1].x;
end Foo;
