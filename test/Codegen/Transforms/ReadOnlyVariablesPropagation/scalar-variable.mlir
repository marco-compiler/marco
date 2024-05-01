// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(propagate-read-only-variables{model-name=Test})" | FileCheck %s

// Propagated scalar constant.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.equation {
// CHECK-NEXT:      %[[lhsValue:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT:      %[[rhsValue:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[lhs:.*]] = bmodelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = bmodelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, constant>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.binding_equation @x {
        %0 = bmodelica.constant #bmodelica.int<0>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.main_model {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.int
            %1 = bmodelica.variable_get @y : !bmodelica.int
            %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
            bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
        }
    }
}

// -----

// Propagated scalar parameter.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.equation {
// CHECK-NEXT:      %[[lhsValue:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT:      %[[rhsValue:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[lhs:.*]] = bmodelica.equation_side %[[lhsValue]]
// CHECK-NEXT:      %[[rhs:.*]] = bmodelica.equation_side %[[rhsValue]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, parameter>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.binding_equation @x {
        %0 = bmodelica.constant #bmodelica.int<0>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.main_model {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.int
            %1 = bmodelica.variable_get @y : !bmodelica.int
            %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
            bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
        }
    }
}
