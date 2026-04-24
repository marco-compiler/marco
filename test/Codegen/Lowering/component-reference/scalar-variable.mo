// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Foo
// CHECK:       bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable.get @x : !bmodelica.real
// CHECK-NEXT:      bmodelica.variable.set @y, %[[x]]
// CHECK-NEXT:  }

function Foo
    input Real x;
    output Real y;
algorithm
    y := x;
end Foo;
