// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<5>
// CHECK-NEXT:      modelica.yield %[[value]] : !modelica.int
// CHECK-NEXT:  } {each = true, fixed = false}

model Test
    Real[3] x(each start = 5);
end Test;
