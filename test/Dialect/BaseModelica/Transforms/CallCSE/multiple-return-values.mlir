//RUN: modelica-opt %s --call-cse | FileCheck %s

// CHECK-LABEL: @Test
module @Test {
    bmodelica.function @foo {
        bmodelica.variable @in : !bmodelica.variable<f64, input>
        bmodelica.variable @out1 : !bmodelica.variable<f64, output>
        bmodelica.variable @out2 : !bmodelica.variable<f32, output>

        bmodelica.algorithm {
            %0 = bmodelica.variable_get @in : f64
            %1 = bmodelica.constant 1.0 : f32
            bmodelica.variable_set @out1, %0 : f64
            bmodelica.variable_set @out2, %1 : f32
        }
    }

    // CHECK: bmodelica.model
    bmodelica.model @M {
        // CHECK-NEXT: bmodelica.variable @[[CSE1:.*]] : !bmodelica.variable<f32>
        // CHECK-NEXT: bmodelica.variable @[[CSE0:.*]] : !bmodelica.variable<f64>
        // CHECK-NEXT: bmodelica.variable @x
        // CHECK-NEXT: bmodelica.variable @y
        bmodelica.variable @x : !bmodelica.variable<f64>
        bmodelica.variable @y : !bmodelica.variable<f32>

        %t0 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @x : f64
            %lhs = bmodelica.equation_side %0 : tuple<f64>
            %1 = bmodelica.constant 1.0 : f64
            // CHECK: %[[RES0:.*]] = bmodelica.variable_get @x
            // CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[RES0]]

            // CHECK: %[[RES1:.*]] = bmodelica.variable_get @[[CSE0]]
            // CHECK: %[[RHS:.*]] = bmodelica.equation_side %[[RES1]]
            // CHECK-NEXT: bmodelica.equation_sides %[[LHS]], %[[RHS]]
            %2:2 = bmodelica.call @foo(%1) : (f64) -> (f64, f32)
            %rhs = bmodelica.equation_side %2#0 : tuple<f64>
            bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
        }

        %t1 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @y : f32
            %lhs = bmodelica.equation_side %0 : tuple<f32>
            %1 = bmodelica.constant 1.0 : f64
            // CHECK: %[[RES0:.*]] = bmodelica.variable_get @y
            // CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[RES0]]

            // CHECK: %[[RES1:.*]] = bmodelica.variable_get @[[CSE1]]
            // CHECK: %[[RHS:.*]] = bmodelica.equation_side %[[RES1]]
            // CHECK-NEXT: bmodelica.equation_sides %[[LHS]], %[[RHS]]
            %2:2 = bmodelica.call @foo(%1) : (f64) -> (f64, f32)
            %rhs = bmodelica.equation_side %2#1 : tuple<f32>
            bmodelica.equation_sides %lhs, %rhs : tuple<f32>, tuple<f32>
        }

        bmodelica.dynamic {
            bmodelica.equation_instance %t0 : !bmodelica.equation
            bmodelica.equation_instance %t1 : !bmodelica.equation
        }

        // CHECK:      %[[TEMPLATE1:.*]] = bmodelica.equation_template inductions = []
        // CHECK-NEXT:     %[[RES2:.*]] = bmodelica.variable_get @[[CSE0]]
        // CHECK-NEXT:     %[[LHS2:.*]] = bmodelica.equation_side %[[RES2]]
        // CHECK-NEXT:     %[[RES3:.*]] = bmodelica.constant 1
        // CHECK-NEXT:     %[[RES4:.*]]:2 = bmodelica.call @foo(%[[RES3]])
        // CHECK-NEXT:     %[[RHS0:.*]] = bmodelica.equation_side %[[RES4]]#0
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS2]], %[[RHS0]]

        // CHECK:      %[[TEMPLATE2:.*]] = bmodelica.equation_template inductions = []
        // CHECK-NEXT:     %[[RES2:.*]] = bmodelica.variable_get @[[CSE1]]
        // CHECK-NEXT:     %[[LHS2:.*]] = bmodelica.equation_side %[[RES2]]
        // CHECK-NEXT:     %[[RES3:.*]] = bmodelica.constant 1
        // CHECK-NEXT:     %[[RES4:.*]]:2 = bmodelica.call @foo(%[[RES3]])
        // CHECK-NEXT:     %[[RHS0:.*]] = bmodelica.equation_side %[[RES4]]#1
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS2]], %[[RHS0]]

        // CHECK:      bmodelica.dynamic
        // CHECK-NEXT:     bmodelica.equation_instance %[[TEMPLATE1]]
        // CHECK-NEXT:     bmodelica.equation_instance %[[TEMPLATE2]]
    }
}
