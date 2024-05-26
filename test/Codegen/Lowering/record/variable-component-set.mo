// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK:               %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.real
// CHECK-NEXT:          bmodelica.variable_component_set %[[x]], @r::@x[] : !bmodelica.real
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real x;
end R;

function Foo
    input Real x;
    output R r;
algorithm
    r.x := x;
end Foo;
