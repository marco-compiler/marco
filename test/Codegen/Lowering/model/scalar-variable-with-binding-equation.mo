// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       bmodelica.binding_equation @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<int 5>
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  }

model Test
    Integer x = 5;
end Test;
