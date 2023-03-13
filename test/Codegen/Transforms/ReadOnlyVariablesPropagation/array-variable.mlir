// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(propagate-read-only-variables{model-name=Test})" | FileCheck %s

// Propagated array constant.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[el0:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[el1:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[el2:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_from_elements %[[el0]], %[[el1]], %[[el2]]
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.load %[[array]][%[[index:.*]]]
// CHECK-NEXT:      %[[y:.*]] = modelica.variable_get @y
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.load %[[y]][%[[index]]]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int, constant>
    modelica.variable @y : !modelica.variable<3x!modelica.int>

    modelica.binding_equation @x {
        %0 = modelica.constant #modelica.int<0>
        %1 = modelica.constant #modelica.int<1>
        %2 = modelica.constant #modelica.int<2>
        %3 = modelica.array_from_elements %0, %1, %2 : !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x!modelica.int>
        modelica.yield %3 : !modelica.array<3x!modelica.int>
    }

    modelica.for_equation %i = 0 to 2 {
        modelica.equation {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.equation_side %1 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}

// -----

// Propagated array parameter.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-NEXT:      %[[el0:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[el1:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:      %[[el2:.*]] = modelica.constant #modelica.int<2>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_from_elements %[[el0]], %[[el1]], %[[el2]]
// CHECK-NEXT:      %[[lhsValue:.*]] = modelica.load %[[array]][%[[index:.*]]]
// CHECK-NEXT:      %[[y:.*]] = modelica.variable_get @y
// CHECK-NEXT:      %[[rhsValue:.*]] = modelica.load %[[y]][%[[index]]]
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int, parameter>
    modelica.variable @y : !modelica.variable<3x!modelica.int>

    modelica.binding_equation @x {
        %0 = modelica.constant #modelica.int<0>
        %1 = modelica.constant #modelica.int<1>
        %2 = modelica.constant #modelica.int<2>
        %3 = modelica.array_from_elements %0, %1, %2 : !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x!modelica.int>
        modelica.yield %3 : !modelica.array<3x!modelica.int>
    }

    modelica.for_equation %i = 0 to 2 {
        modelica.equation {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.variable_get @y : !modelica.array<3x!modelica.int>
            %3 = modelica.load %2[%i] : !modelica.array<3x!modelica.int>
            %4 = modelica.equation_side %1 : tuple<!modelica.int>
            %5 = modelica.equation_side %3 : tuple<!modelica.int>
            modelica.equation_sides %4, %5 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}
