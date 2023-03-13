// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(propagate-read-only-variables{model-name=Test ignored-variables="x,y"})" | FileCheck %s

// Multiple variables set as not to be propagated.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[xValue:.*]] = modelica.variable_get @x
// CHECK-NEXT:      %[[yValue:.*]] = modelica.variable_get @y
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.add %[[xValue]], %[[yValue]]
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.variable_get @z
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int, constant>
    modelica.variable @y : !modelica.variable<!modelica.int, parameter>
    modelica.variable @z : !modelica.variable<!modelica.int>

    modelica.binding_equation @x {
        %0 = modelica.constant #modelica.int<0>
        modelica.yield %0 : !modelica.int
    }

    modelica.binding_equation @y {
        %0 = modelica.constant #modelica.int<1>
        modelica.yield %0 : !modelica.int
    }

    modelica.equation {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.add %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        %3 = modelica.variable_get @z : !modelica.int
        %4 = modelica.equation_side %2 : tuple<!modelica.int>
        %5 = modelica.equation_side %3 : tuple<!modelica.int>
        modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
