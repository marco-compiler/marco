// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK:       modelica.model @Test {
// CHECK-NEXT:  }

model Test
end Test;
