// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK:               %[[r:.*]] = bmodelica.variable_get @r : tensor<2x!bmodelica<record @R>>
// CHECK:               %[[r_x:.*]] = bmodelica.component_get %[[r]], @x : tensor<2x!bmodelica<record @R>> -> tensor<2x3x!bmodelica.real>
// CHECK-NEXT:          bmodelica.variable_set @x, %[[r_x]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real[3] x;
end R;

function Foo
    input R[2] r;
    output Real[2,3] x;
algorithm
    x := r.x;
end Foo;
