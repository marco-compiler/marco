// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.model @Test {
// CHECK-NEXT:  }

model Test
end Test;
