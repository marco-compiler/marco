// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK:       modelica.record @Test {
// CHECK-NEXT:  }

record Test
end Test;
