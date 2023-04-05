// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK:       modelica.record @Test {
// CHECK-NEXT:  }

record Test
end Test;
