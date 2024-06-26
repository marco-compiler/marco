// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK-NEXT:          %[[r:.*]] = bmodelica.variable_get @r : !bmodelica<record @R>
// CHECK-NEXT:          %[[r_x:.*]] = bmodelica.component_get %[[r]], @x : !bmodelica<record @R> -> !bmodelica.real
// CHECK-NEXT:          bmodelica.variable_set @x, %[[r_x]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real x;
end R;

function Foo
    input R r;
    output Real x;
algorithm
    x := r.x;
end Foo;
