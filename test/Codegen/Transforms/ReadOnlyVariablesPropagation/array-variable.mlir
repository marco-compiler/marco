// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(propagate-read-only-variables{model-name=Test})" | FileCheck %s

// Propagated array constant.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.equation {
// CHECK-NEXT:      %[[el0:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT:      %[[el1:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-NEXT:      %[[el2:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-NEXT:      %[[array:.*]] = bmodelica.array_from_elements %[[el0]], %[[el1]], %[[el2]]
// CHECK-NEXT:      %[[lhsValue:.*]] = bmodelica.load %[[array]][%[[index:.*]]]
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[rhsValue:.*]] = bmodelica.load %[[y]][%[[index]]]
// CHECK-NEXT:      %[[lhs:.*]] = bmodelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = bmodelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, constant>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>

    bmodelica.binding_equation @x {
        %0 = bmodelica.constant #bmodelica.int<0>
        %1 = bmodelica.constant #bmodelica.int<1>
        %2 = bmodelica.constant #bmodelica.int<2>
        %3 = bmodelica.array_from_elements %0, %1, %2 : !bmodelica.int, !bmodelica.int, !bmodelica.int -> !bmodelica.array<3x!bmodelica.int>
        bmodelica.yield %3 : !bmodelica.array<3x!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.for_equation %i = 0 to 2 {
            bmodelica.equation {
                %0 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
                %1 = bmodelica.load %0[%i] : !bmodelica.array<3x!bmodelica.int>
                %2 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
                %3 = bmodelica.load %2[%i] : !bmodelica.array<3x!bmodelica.int>
                %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
                %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
                bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}

// -----

// Propagated array parameter.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.equation {
// CHECK-NEXT:      %[[el0:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT:      %[[el1:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-NEXT:      %[[el2:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-NEXT:      %[[array:.*]] = bmodelica.array_from_elements %[[el0]], %[[el1]], %[[el2]]
// CHECK-NEXT:      %[[lhsValue:.*]] = bmodelica.load %[[array]][%[[index:.*]]]
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[rhsValue:.*]] = bmodelica.load %[[y]][%[[index]]]
// CHECK-NEXT:      %[[lhs:.*]] = bmodelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = bmodelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>

    bmodelica.binding_equation @x {
        %0 = bmodelica.constant #bmodelica.int<0>
        %1 = bmodelica.constant #bmodelica.int<1>
        %2 = bmodelica.constant #bmodelica.int<2>
        %3 = bmodelica.array_from_elements %0, %1, %2 : !bmodelica.int, !bmodelica.int, !bmodelica.int -> !bmodelica.array<3x!bmodelica.int>
        bmodelica.yield %3 : !bmodelica.array<3x!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.for_equation %i = 0 to 2 {
            bmodelica.equation {
                %0 = bmodelica.variable_get @x : !bmodelica.array<3x!bmodelica.int>
                %1 = bmodelica.load %0[%i] : !bmodelica.array<3x!bmodelica.int>
                %2 = bmodelica.variable_get @y : !bmodelica.array<3x!bmodelica.int>
                %3 = bmodelica.load %2[%i] : !bmodelica.array<3x!bmodelica.int>
                %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
                %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
                bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}
