// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(propagate-read-only-variables{model-name=Test ignored-variables="x,y"})" | FileCheck %s

// Multiple variables set as not to be propagated.

// CHECK-LABEL: @Test
// CHECK: ^bb0(%[[x:.*]]: !modelica.array<!modelica.int>, %[[y:.*]]: !modelica.array<!modelica.int>, %[[z:.*]]: !modelica.array<!modelica.int>):
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[xValue:.*]] = modelica.load %[[x]][]
// CHECK-NEXT:      %[[yValue:.*]] = modelica.load %[[y]][]
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.add %[[xValue]], %[[yValue]]
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.load %[[z]][]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, constant>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int, parameter>
    %2 = modelica.member_create @z : !modelica.member<!modelica.int>
    modelica.yield %0, %1, %2 : !modelica.member<!modelica.int, constant>, !modelica.member<!modelica.int, parameter>, !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>, %arg1: !modelica.array<!modelica.int>, %arg2: !modelica.array<!modelica.int>):
    modelica.binding_equation (%arg0 : !modelica.array<!modelica.int>) {
        %0 = modelica.constant #modelica.int<0>
        modelica.yield %0 : !modelica.int
    }

    modelica.binding_equation (%arg1 : !modelica.array<!modelica.int>) {
        %0 = modelica.constant #modelica.int<1>
        modelica.yield %0 : !modelica.int
    }

    modelica.equation {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.load %arg1[] : !modelica.array<!modelica.int>
        %2 = modelica.add %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        %3 = modelica.load %arg2[] : !modelica.array<!modelica.int>
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
