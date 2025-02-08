// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(propagate-read-only-variables{model-name=Test})" | FileCheck %s

// CHECK-LABEL: @propagatedScalarConstant

bmodelica.model @propagatedScalarConstant {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, constant>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.binding_equation @x {
        %0 = bmodelica.constant #bmodelica<int 0>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.int
            %1 = bmodelica.variable_get @y : !bmodelica.int
            %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
            bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
        }
    }

    // CHECK:       bmodelica.equation
    // CHECK-DAG:       %[[lhsValue:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-DAG:       %[[rhsValue:.*]] = bmodelica.variable_get @y
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[lhsValue]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[rhsValue]]
    // CHECK-DAG:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
}

// -----

// CHECK-LABEL: @propagatedScalarParameter

bmodelica.model @propagatedScalarParameter {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, parameter>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.binding_equation @x {
        %0 = bmodelica.constant #bmodelica<int 0>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.int
            %1 = bmodelica.variable_get @y : !bmodelica.int
            %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
            bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
        }
    }

    // CHECK:       bmodelica.equation
    // CHECK-DAG:       %[[lhsValue:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-DAG:       %[[rhsValue:.*]] = bmodelica.variable_get @y
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[lhsValue]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[rhsValue]]
    // CHECK-DAG:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
}
