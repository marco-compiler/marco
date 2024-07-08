// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.record @Test {
// CHECK-NEXT:      bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool>
// CHECK-NEXT:      bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>
// CHECK-NEXT:      bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  }

record Test
    Boolean x;
    Integer y;
    Real z;
end Test;
