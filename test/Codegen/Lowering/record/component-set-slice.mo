// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK-DAG:           %[[r:.*]] = bmodelica.variable_get @r : !bmodelica.array<3x!bmodelica<record @R>>
// CHECK-DAG:           %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.real>
// CHECK-NEXT:          bmodelica.component_set %[[r]], @x, %[[x]] : !bmodelica.array<3x!bmodelica<record @R>>, !bmodelica.array<3x!bmodelica.real>
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real x;
end R;

function Foo
    input Real[3] x;
    output R[3] r;
algorithm
    r.x := x;
end Foo;
