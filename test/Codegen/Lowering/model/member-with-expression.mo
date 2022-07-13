// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: body
// CHECK-NEXT:  ^bb0(%[[X:[a-zA-Z0-9]*]]: !modelica.array<!modelica.int>):
// CHECK-NEXT:      modelica.equation {
// CHECK-NEXT:          %[[X_LOAD:[a-zA-Z0-9]*]] = modelica.load %[[X]][] : !modelica.array<!modelica.int>
// CHECK-NEXT:          %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:          %[[LHS:[a-zA-Z0-9]*]] = modelica.equation_side %[[X_LOAD]] : tuple<!modelica.int>
// CHECK-NEXT:          %[[RHS:[a-zA-Z0-9]*]] = modelica.equation_side %[[VALUE]] : tuple<!modelica.int>
// CHECK-NEXT:          modelica.equation_sides %[[LHS]], %[[RHS]] : tuple<!modelica.int>, tuple<!modelica.int>
// CHECK-NEXT:      }

model MemberWithExpression
    Integer x = 0;
end MemberWithExpression;
