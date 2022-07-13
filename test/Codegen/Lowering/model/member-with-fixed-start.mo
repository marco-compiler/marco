// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: body
// CHECK-NEXT:  ^bb0(%[[n:[a-zA-Z0-9]*]]: !modelica.array<!modelica.real>):
// CHECK-NEXT:      modelica.start (%[[n]] : !modelica.array<!modelica.real>) {each = false, fixed = true} {
// CHECK-NEXT:          %[[value:[a-zA-Z0-9]*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT:          modelica.yield %[[value]] : !modelica.real
// CHECK-NEXT:      }

model MemberWithFixedStart
    Real x(start = 5, fixed = true);
end MemberWithFixedStart;
