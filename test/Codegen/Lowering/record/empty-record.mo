// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.record @Test {
// CHECK-NEXT:  }

record Test
end Test;
