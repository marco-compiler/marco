// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<5>
// CHECK-NEXT:      modelica.yield %[[value]] : !modelica.int
// CHECK-NEXT:  } {each = false, fixed = true}

model Test
    Real x(start = 5, fixed = true);
end Test;
