// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[n:.*]]: !modelica.array<!modelica.real>):
// CHECK-NEXT:      modelica.start (%[[n]] : !modelica.array<!modelica.real>) {each = false, fixed = false} {
// CHECK-NEXT:          %[[value:.*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT:          modelica.yield %[[value]] : !modelica.real
// CHECK-NEXT:      }

model Test
    Real x(start = 5, fixed = false);
end Test;
