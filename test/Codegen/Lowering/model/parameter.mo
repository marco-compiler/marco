// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: body
// CHECK-NEXT:  ^bb0(%[[n:[a-zA-Z0-9]*]]: !modelica.array<!modelica.int>):
// CHECK-NEXT:      modelica.start (%[[n]] : !modelica.array<!modelica.int>) {
// CHECK-NEXT:          %[[value:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<5>
// CHECK-NEXT:          modelica.yield %[[value]] : !modelica.int
// CHECK-NEXT:      }

model ModelWithParameter
    parameter Integer n = 5;
end ModelWithParameter;
