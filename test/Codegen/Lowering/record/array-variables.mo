// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK:       modelica.record @Test {
// CHECK-NEXT:      modelica.variable @x : !modelica.variable<3x!modelica.bool>
// CHECK-NEXT:      modelica.variable @y : !modelica.variable<4x!modelica.int>
// CHECK-NEXT:      modelica.variable @z : !modelica.variable<5x!modelica.real>
// CHECK-NEXT:  }

record Test
    Boolean x[3];
    Integer y[4];
    Real z[5];
end Test;
