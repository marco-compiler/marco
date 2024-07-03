// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.record @Test {
// CHECK-NEXT:      bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.bool>
// CHECK-NEXT:      bmodelica.variable @y : !bmodelica.variable<4x!bmodelica.int>
// CHECK-NEXT:      bmodelica.variable @z : !bmodelica.variable<5x!bmodelica.real>
// CHECK-NEXT:  }

record Test
    Boolean[3] x;
    Integer[4] y;
    Real[5] z;
end Test;
