// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(propagate-parameters{model-name=Test})" | FileCheck %s

// Propagated array parameter.

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[x:.*]]: !modelica.array<3x!modelica.int>, %[[y:.*]]: !modelica.array<3x!modelica.int>):
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[el0:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[el1:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[el2:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_from_elements %[[el0]], %[[el1]], %[[el2]]
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.load %[[array]][%[[index:.*]]]
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.load %[[y]][%[[index]]]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int, parameter>
    %1 = modelica.member_create @y : !modelica.member<3x!modelica.int>
    modelica.yield %0, %1 : !modelica.member<3x!modelica.int, parameter>, !modelica.member<3x!modelica.int>
} body {
^bb0(%arg0: !modelica.array<3x!modelica.int>, %arg1: !modelica.array<3x!modelica.int>):
    modelica.binding_equation (%arg0 : !modelica.array<3x!modelica.int>) {
        %0 = modelica.constant #modelica.int<0>
        %1 = modelica.constant #modelica.int<1>
        %2 = modelica.constant #modelica.int<2>
        %3 = modelica.array_from_elements %0, %1, %2 : !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x!modelica.int>
        modelica.yield %3 : !modelica.array<3x!modelica.int>
    }

    modelica.for_equation %i = 0 to 2 {
        modelica.equation {
            %0 = modelica.load %arg0[%i] : !modelica.array<3x!modelica.int>
            %1 = modelica.load %arg1[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.equation_side %0 : tuple<!modelica.int>
            %3 = modelica.equation_side %1 : tuple<!modelica.int>
            modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}
