// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[x:.*]]: !modelica.array<!modelica.int>):
// CHECK-NEXT:      modelica.binding_equation (%[[x]] : !modelica.array<!modelica.int>) {
// CHECK-NEXT:          %[[value:.*]] = modelica.constant #modelica.int<5>
// CHECK-NEXT:          modelica.yield %[[value]]
// CHECK-NEXT:      }

model Test
    Integer x = 5;
end Test;
