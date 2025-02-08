// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(propagate-read-only-variables{model-name=Test ignored-variables="x,y"})" | FileCheck %s

// COM: Multiple variables set as not to be propagated.

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, constant>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, parameter>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.int>

    bmodelica.binding_equation @x {
        %0 = bmodelica.constant #bmodelica<int 0>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.binding_equation @y {
        %0 = bmodelica.constant #bmodelica<int 1>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.int
            %1 = bmodelica.variable_get @y : !bmodelica.int
            %2 = bmodelica.add %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
            %3 = bmodelica.variable_get @z : !bmodelica.int
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
            bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
        }
    }

    // CHECK:       bmodelica.equation
    // CHECK-DAG:       %[[xValue:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[yValue:.*]] = bmodelica.variable_get @y
    // CHECK-DAG:       %[[lhsValue:.*]] = bmodelica.add %[[xValue]], %[[yValue]]
    // CHECK-DAG:       %[[rhsValue:.*]] = bmodelica.variable_get @z
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[lhsValue]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[rhsValue]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
}
