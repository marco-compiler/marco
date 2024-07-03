// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK:               %[[x:.*]] = bmodelica.variable_get @x : tensor<2x3x!bmodelica.real>
// CHECK-NEXT:          bmodelica.variable_component_set @r::@x, %[[x]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real[3] x;
end R;

function Foo
    input Real[2,3] x;
    output R[2] r;
algorithm
    r.x := x;
end Foo;
