// RUN: modelica-opt %s --split-input-file --call-cse | FileCheck %s

// CHECK-LABEL: @SingleCall
module @SingleCall {
    bmodelica.function @foo {
        bmodelica.variable @x : !bmodelica.variable<f64, input>
        bmodelica.variable @y : !bmodelica.variable<f64, output>

        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : f64
            bmodelica.variable_set @y, %0 : f64
        }
    }

    bmodelica.model @M {
        bmodelica.variable @x : !bmodelica.variable<f64>

        // CHECK: bmodelica.equation_template
        // CHECK-NEXT: %[[R0:.*]] = bmodelica.variable_get @x
        // CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[R0]]
        // CHECK-NEXT: %[[R1:.*]] = bmodelica.constant 1
        // CHECK-NEXT: %[[R2:.*]] = bmodelica.call @foo(%[[R1]])
        // CHECK-NEXT: %[[RHS:.*]] = bmodelica.equation_side %[[R2]]
        // CHECK-NEXT: bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %t0 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @x : f64
            %lhs = bmodelica.equation_side %0 : tuple<f64>
            %1 = bmodelica.constant 1.0 : f64
            %2 = bmodelica.call @foo(%1) : (f64) -> f64
            %rhs = bmodelica.equation_side %2 : tuple<f64>
            bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
        }

        bmodelica.dynamic {
            bmodelica.equation_instance %t0 : !bmodelica.equation
        }
    }
}

// -----

// CHECK-LABEL: @InductionVariables
module @InductionVariables {
    bmodelica.function @foo {
        bmodelica.variable @x : !bmodelica.variable<f64, input>
        bmodelica.variable @y : !bmodelica.variable<f64, output>

        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : f64
            bmodelica.variable_set @y, %0 : f64
        }
    }

    bmodelica.model @M {
        bmodelica.variable @x : !bmodelica.variable<f64>
        bmodelica.variable @y : !bmodelica.variable<f64>

        // CHECK:      bmodelica.equation_template
        // CHECK-NEXT:     %[[R0:.*]] = bmodelica.variable_get @x
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[R0]]
        // CHECK-NEXT:     %[[R1:.*]] = bmodelica.constant 1
        // CHECK-NEXT:     %[[R2:.*]] = bmodelica.call @foo(%[[R1]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[R2]]
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %t0 = bmodelica.equation_template inductions = [%i1] {
            %0 = bmodelica.variable_get @x : f64
            %lhs = bmodelica.equation_side %0 : tuple<f64>
            %1 = bmodelica.constant 1.0 : f64
            %2 = bmodelica.call @foo(%1) : (f64) -> f64
            %rhs = bmodelica.equation_side %2 : tuple<f64>
            bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
        }

        // CHECK:      bmodelica.equation_template
        // CHECK-NEXT:     %[[R0:.*]] = bmodelica.variable_get @x
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[R0]]
        // CHECK-NEXT:     %[[R1:.*]] = bmodelica.constant 1
        // CHECK-NEXT:     %[[R2:.*]] = bmodelica.call @foo(%[[R1]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[R2]]
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %t1 = bmodelica.equation_template inductions = [%i1] {
            %0 = bmodelica.variable_get @x : f64
            %lhs = bmodelica.equation_side %0 : tuple<f64>
            %1 = bmodelica.constant 1.0 : f64
            %2 = bmodelica.call @foo(%1) : (f64) -> f64
            %rhs = bmodelica.equation_side %2 : tuple<f64>
            bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
        }

        bmodelica.dynamic {
            bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [0, 1]>} : !bmodelica.equation
            bmodelica.equation_instance %t1 {indices = #modeling<multidim_range [0, 1]>} : !bmodelica.equation
        }
    }
}