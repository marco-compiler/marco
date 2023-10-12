// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       modelica.record @Test {
// CHECK-NEXT:      modelica.variable @x : !modelica.variable<!modelica.bool>
// CHECK-NEXT:      modelica.variable @y : !modelica.variable<!modelica.int>
// CHECK-NEXT:      modelica.variable @z : !modelica.variable<!modelica.real>
// CHECK-NEXT:  }

record Test
    Boolean x;
    Integer y;
    Real z;
end Test;
