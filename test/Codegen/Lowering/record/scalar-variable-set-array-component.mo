// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK:               %[[x:.*]] = bmodelica.variable_get @x : tensor<3x!bmodelica.real>
// CHECK-NEXT:          bmodelica.variable_component_set @r::@x, %[[x]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real[3] x;
end R;

function Foo
    input Real[3] x;
    output R r;
algorithm
    r.x := x;
end Foo;
