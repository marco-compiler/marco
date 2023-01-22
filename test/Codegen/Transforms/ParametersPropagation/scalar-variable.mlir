// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(propagate-parameters{model-name=Test})" | FileCheck %s

// Propagated scalar parameter.

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[x:.*]]: !modelica.array<!modelica.int>, %[[y:.*]]: !modelica.array<!modelica.int>):
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.load %[[y]][]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, parameter>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int>
    modelica.yield %0, %1 : !modelica.member<!modelica.int, parameter>, !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>, %arg1: !modelica.array<!modelica.int>):
    modelica.binding_equation (%arg0 : !modelica.array<!modelica.int>) {
        %0 = modelica.constant #modelica.int<0>
        modelica.yield %0 : !modelica.int
    }

    modelica.equation {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}