// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Foo
// CHECK:       bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
// CHECK-NEXT:      bmodelica.variable_set @y, %[[x]]
// CHECK-NEXT:  }

function Foo
    input Real[2] x;
    output Real[2] y;
algorithm
    y := x;
end Foo;
