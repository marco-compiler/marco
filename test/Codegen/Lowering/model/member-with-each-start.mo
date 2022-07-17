// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: body
// CHECK-NEXT:  ^bb0(%[[n:.*]]: !modelica.array<3x!modelica.real>):
// CHECK-NEXT:      modelica.start (%[[n]] : !modelica.array<3x!modelica.real>) {each = true, fixed = false} {
// CHECK-NEXT:          %[[value:.*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT:          modelica.yield %[[value]] : !modelica.real
// CHECK-NEXT:      }

model MemberWithEachStart
    Real[3] x(each start = 5);
end MemberWithEachStart;
