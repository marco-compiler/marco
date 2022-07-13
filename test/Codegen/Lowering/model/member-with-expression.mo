// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: body
// CHECK-NEXT:  ^bb0(%[[x:[a-zA-Z0-9]*]]: !modelica.array<!modelica.int>):
// CHECK-NEXT:      modelica.equation {
// CHECK-NEXT:          %[[x_load:[a-zA-Z0-9]*]] = modelica.load %[[x]][] : !modelica.array<!modelica.int>
// CHECK-NEXT:          %[[value:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<5>
// CHECK-DAG:           %[[lhs:[a-zA-Z0-9]*]] = modelica.equation_side %[[x_load]] : tuple<!modelica.int>
// CHECK-DAG:           %[[rhs:[a-zA-Z0-9]*]] = modelica.equation_side %[[value]] : tuple<!modelica.int>
// CHECK-NEXT:          modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:      }

model MemberWithExpression
    Integer x = 5;
end MemberWithExpression;
