// RUN: modelica-opt %s --call-cse | FileCheck %s

// CHECK-LABEL: @Test
module @Test {
    bmodelica.function @foo {
        bmodelica.variable @x : !bmodelica.variable<f64, input>
        bmodelica.variable @y : !bmodelica.variable<f64, output>

        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : f64
            bmodelica.variable_set @y, %0 : f64
        }
    }

    // CHECK-LABEL: @M
    bmodelica.model @M {
        // CHECK-NEXT: bmodelica.variable @[[CSE:.*]] : !bmodelica.variable<f64>
        // CHECK-NEXT: bmodelica.variable @x
        // CHECK-NEXT: bmodelica.variable @y
        bmodelica.variable @x : !bmodelica.variable<f64>
        bmodelica.variable @y : !bmodelica.variable<f64>

        %t0 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @x : f64
            %lhs = bmodelica.equation_side %0 : tuple<f64>
            %1 = bmodelica.constant 1.0 : f64
            // CHECK: %[[RES0:.*]] = bmodelica.variable_get @x
            // CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[RES0]]

            // CHECK: %[[RES1:.*]] = bmodelica.variable_get @[[CSE]]
            // CHECK-NEXT: %[[RHS:.*]] = bmodelica.equation_side %[[RES1]]
            // CHECK-NEXT: bmodelica.equation_sides %[[LHS]], %[[RHS]]
            %2 = bmodelica.call @foo(%1) : (f64) -> f64
            %rhs = bmodelica.equation_side %2 : tuple<f64>
            bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
        }

        %t1 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @y : f64
            %lhs = bmodelica.equation_side %0 : tuple<f64>
            %1 = bmodelica.constant 1.0 : f64
            // CHECK: %[[RES0:.*]] = bmodelica.variable_get @y
            // CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[RES0]]

            // CHECK: %[[RES1:.*]] = bmodelica.variable_get @[[CSE]]
            // CHECK-NEXT: %[[RHS:.*]] = bmodelica.equation_side %[[RES1]]
            // CHECK-NEXT: bmodelica.equation_sides %[[LHS]], %[[RHS]]
            %2 = bmodelica.call @foo(%1) : (f64) -> f64
            %rhs = bmodelica.equation_side %2 : tuple<f64>
            bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
        }

        bmodelica.dynamic {
            bmodelica.equation_instance %t0 : !bmodelica.equation
            bmodelica.equation_instance %t1 : !bmodelica.equation
        }

        // CHECK:      %[[TEMPLATE:.*]] = bmodelica.equation_template inductions = []
        // CHECK-NEXT:     %[[RES2:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[LHS2:.*]] = bmodelica.equation_side %[[RES2]]
        // CHECK-NEXT:     %[[RES3:.*]] = bmodelica.constant 1
        // CHECK-NEXT:     %[[RES4:.*]] = bmodelica.call @foo(%[[RES3]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[RES4]]
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS2]], %[[RHS]]

        // CHECK:      bmodelica.dynamic
        // CHECK-NEXT:     bmodelica.equation_instance %[[TEMPLATE]]
    }
}