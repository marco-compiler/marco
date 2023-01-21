// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[n:.*]]: !modelica.array<3x!modelica.real>):
// CHECK-NEXT:      modelica.start (%[[n]] : !modelica.array<3x!modelica.real>) {each = true, fixed = false} {
// CHECK-NEXT:          %[[value:.*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT:          modelica.yield %[[value]] : !modelica.real
// CHECK-NEXT:      }

model Test
    Real[3] x(each start = 5);
end Test;
