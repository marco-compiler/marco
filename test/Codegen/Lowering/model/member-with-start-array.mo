// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: body
// CHECK-NEXT:  ^bb0(%[[arg0:.*]]: !modelica.array<3x!modelica.real>):
// CHECK-NEXT:    modelica.equation {
// CHECK-NEXT:      %[[cst0:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[cst1:.*]] = modelica.constant #modelica.real<2.000000e+00>
// CHECK-NEXT:      %[[cst2:.*]] = modelica.constant #modelica.int<3>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_from_elements %[[cst0]], %[[cst1]], %[[cst2]] : !modelica.int, !modelica.real, !modelica.int -> !modelica.array<3x!modelica.real>
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[arg0]] : tuple<!modelica.array<3x!modelica.real>>
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[array]] : tuple<!modelica.array<3x!modelica.real>>
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:    }

model MemberWithStartArray
	Real[3] x = {1, 2.0, 3};
equation
end MemberWithStartArray;
