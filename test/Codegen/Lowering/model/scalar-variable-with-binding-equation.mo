// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       modelica.binding_equation @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<5>
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  }

model Test
    Integer x = 5;
end Test;
