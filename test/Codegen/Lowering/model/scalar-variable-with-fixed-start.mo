// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[n:.*]]: !modelica.array<!modelica.real>):
// CHECK-NEXT:      modelica.start (%[[n]] : !modelica.array<!modelica.real>) {each = false, fixed = true} {
// CHECK-NEXT:          %[[value:.*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT:          modelica.yield %[[value]] : !modelica.real
// CHECK-NEXT:      }

model Test
    Real x(start = 5, fixed = true);
end Test;
