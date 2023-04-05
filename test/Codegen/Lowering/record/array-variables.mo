// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK:       modelica.record @Test {
// CHECK-NEXT:      modelica.variable @x : !modelica.variable<3x!modelica.bool>
// CHECK-NEXT:      modelica.variable @y : !modelica.variable<4x!modelica.int>
// CHECK-NEXT:      modelica.variable @z : !modelica.variable<5x!modelica.real>
// CHECK-NEXT:  }

record Test
    Boolean[3] x;
    Integer[4] y;
    Real[5] z;
end Test;
