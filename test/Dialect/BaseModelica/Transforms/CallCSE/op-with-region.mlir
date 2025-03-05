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
        // CHECK: bmodelica.variable @[[CSE:.*]] : !bmodelica.variable<f64>
        bmodelica.variable @x : !bmodelica.variable<f64>
        bmodelica.variable @y : !bmodelica.variable<f64>

        %t0 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @x : f64
            %lhs = bmodelica.equation_side %0 : tuple<f64>

            %1 = bmodelica.constant 2.0 : f64

            %lower = bmodelica.constant 5 : index
            %upper = bmodelica.constant 10 : index
            %step = bmodelica.constant 1 : index
            %range = bmodelica.range %lower, %upper, %step : (index, index, index) -> !bmodelica<range index>

            %red = bmodelica.reduction "add", iterables = [%range], inductions = [] {
                bmodelica.yield %1 : f64
            } : (!bmodelica<range index>) -> f64

            // CHECK: %[[RES0:.*]] = bmodelica.variable_get @x
            // CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[RES0]]

            // CHECK: %[[RES1:.*]] = bmodelica.variable_get @[[CSE]]
            // CHECK-NEXT: %[[RHS:.*]] = bmodelica.equation_side %[[RES1]]
            // CHECK-NEXT: bmodelica.equation_sides %[[LHS]], %[[RHS]]

            %2 = bmodelica.call @foo(%red) : (f64) -> f64

            %rhs = bmodelica.equation_side %2 : tuple<f64>
            bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
        }

        %t1 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @y : f64
            %lhs = bmodelica.equation_side %0 : tuple<f64>

            %1 = bmodelica.constant 2.0 : f64

            %lower = bmodelica.constant 5 : index
            %upper = bmodelica.constant 10 : index
            %step = bmodelica.constant 1 : index
            %range = bmodelica.range %lower, %upper, %step : (index, index, index) -> !bmodelica<range index>

            %red = bmodelica.reduction "add", iterables = [%range], inductions = [] {
                bmodelica.yield %1 : f64
            } : (!bmodelica<range index>) -> f64

            // CHECK: %[[RES0:.*]] = bmodelica.variable_get @y
            // CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[RES0]]

            // CHECK: %[[RES1:.*]] = bmodelica.variable_get @[[CSE]]
            // CHECK-NEXT: %[[RHS:.*]] = bmodelica.equation_side %[[RES1]]
            // CHECK-NEXT: bmodelica.equation_sides %[[LHS]], %[[RHS]]

            %2 = bmodelica.call @foo(%red) : (f64) -> f64

            %rhs = bmodelica.equation_side %2 : tuple<f64>
            bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
        }

        bmodelica.dynamic {
            bmodelica.equation_instance %t0
            bmodelica.equation_instance %t1
        }

        // CHECK:      %[[TEMPLATE:.*]] = bmodelica.equation_template inductions = []
        // CHECK-NEXT:     %[[RES:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[RES]]
        // CHECK-DAG:      %[[c:.*]] = bmodelica.constant 2
        // CHECK-DAG:      %[[lower:.*]] = bmodelica.constant 5
        // CHECK-DAG:      %[[upper:.*]] = bmodelica.constant 10
        // CHECK-DAG:      %[[step:.*]] = bmodelica.constant 1
        // CHECK-DAG:      %[[range:.*]] = bmodelica.range %[[lower]], %[[upper]], %[[step]]

        // CHECK-NEXT:     %[[red:.*]] = bmodelica.reduction "add", iterables = [%[[range]]]
        // CHECK-NEXT:         bmodelica.yield %[[c]]

        // CHECK:          %[[RES1:.*]] = bmodelica.call @foo(%[[red]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[RES1]]
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]

        // CHECK:      bmodelica.dynamic
        // CHECK-NEXT:     bmodelica.equation_instance %[[TEMPLATE]]
    }
}
