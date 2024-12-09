// RUN: modelica-opt %s --split-input-file --call-cse | FileCheck %s

// CHECK-LABEL: @SingleCall
module @SingleCall {
    bmodelica.function @foo {
        bmodelica.variable @x : !bmodelica.variable<f64, input>
        bmodelica.variable @y : !bmodelica.variable<f64, output>
    }

    bmodelica.model @M {
        bmodelica.variable @x : !bmodelica.variable<f64>

        // CHECK:      %[[T:.*]] = bmodelica.equation_template
        // CHECK-NEXT:     %[[R0:.*]] = bmodelica.variable_get @x
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[R0]]
        // CHECK-NEXT:     %[[R1:.*]] = bmodelica.constant 1
        // CHECK-NEXT:     %[[R2:.*]] = bmodelica.call @foo(%[[R1]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[R2]]
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %t0 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @x : f64
            %lhs = bmodelica.equation_side %0 : tuple<f64>
            %1 = bmodelica.constant 1.0 : f64
            %2 = bmodelica.call @foo(%1) : (f64) -> f64
            %rhs = bmodelica.equation_side %2 : tuple<f64>
            bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
        }

        // CHECK: bmodelica.dynamic
        bmodelica.dynamic {
            // CHECK-DAG: bmodelica.equation_instance %[[T]]
            bmodelica.equation_instance %t0 : !bmodelica.equation
        }
    }
}

// -----

// CHECK-LABEL: @ArrayResult
module @ArrayResult {
    bmodelica.function @FuncWithArrayResult {
        bmodelica.variable @x : !bmodelica.variable<f64, input>
        bmodelica.variable @y : !bmodelica.variable<1xf64, output>
    }

    bmodelica.model @M  {
        bmodelica.variable @x : !bmodelica.variable<1xf64>
        bmodelica.variable @y : !bmodelica.variable<1xf64>

        // CHECK:      %[[T0:.*]] = bmodelica.equation_template
        // CHECK-NEXT:     %[[R0:.*]] = bmodelica.variable_get @x
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[R0]]
        // CHECK-NEXT:     %[[R1:.*]] = bmodelica.constant 1
        // CHECK-NEXT:     %[[R2:.*]] = bmodelica.call @FuncWithArrayResult(%[[R1]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[R2]]
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %t0 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @x : tensor<1xf64>
            %1 = bmodelica.equation_side %0 : tuple<tensor<1xf64>>
            %2 = bmodelica.constant 1.0 : f64
            %3 = bmodelica.call @FuncWithArrayResult(%2) : (f64) -> tensor<1xf64>
            %4 = bmodelica.equation_side %3 : tuple<tensor<1xf64>>
            bmodelica.equation_sides %1, %4 : tuple<tensor<1xf64>>, tuple<tensor<1xf64>>
        }

        // CHECK:      %[[T1:.*]] = bmodelica.equation_template
        // CHECK-NEXT:     %[[R0:.*]] = bmodelica.variable_get @y
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[R0]]
        // CHECK-NEXT:     %[[R1:.*]] = bmodelica.constant 1
        // CHECK-NEXT:     %[[R2:.*]] = bmodelica.call @FuncWithArrayResult(%[[R1]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[R2]]
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %t1 = bmodelica.equation_template inductions = [] {
            %0 = bmodelica.variable_get @y : tensor<1xf64>
            %1 = bmodelica.equation_side %0 : tuple<tensor<1xf64>>
            %2 = bmodelica.constant 1.0 : f64
            %3 = bmodelica.call @FuncWithArrayResult(%2) : (f64) -> tensor<1xf64>
            %4 = bmodelica.equation_side %3 : tuple<tensor<1xf64>>
            bmodelica.equation_sides %1, %4 : tuple<tensor<1xf64>>, tuple<tensor<1xf64>>
        }

        // CHECK: bmodelica.dynamic
        bmodelica.dynamic {
            // CHECK-DAG: bmodelica.equation_instance %[[T0]]
            // CHECK-DAG: bmodelica.equation_instance %[[T1]]
            bmodelica.equation_instance %t0 : !bmodelica.equation
            bmodelica.equation_instance %t1 : !bmodelica.equation
        }
    }
}

// -----

// CHECK-LABEL: @ConflictingIndices
module @ConflictingIndices {
    bmodelica.function @f {
        bmodelica.variable @x : !bmodelica.variable<i32, input>
        bmodelica.variable @y : !bmodelica.variable<i32, input>
        bmodelica.variable @z : !bmodelica.variable<i32, output>
    }

    // CHECK-LABEL: @M
    bmodelica.model @M  {
        // CHECK-NEXT: bmodelica.variable @a
        // CHECK-NEXT: bmodelica.variable @b
        bmodelica.variable @a : !bmodelica.variable<4x5xi32>
        bmodelica.variable @b : !bmodelica.variable<4x5xi32>

        // COM: This template is instantiated with different indices, so it should be kept as is.

        // CHECK:      %[[T0:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[OFFSET:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET]]
        // CHECK-NEXT:     %[[INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET]]
        // CHECK-NEXT:     %[[ARR:.*]] = bmodelica.variable_get @a
        // CHECK-NEXT:     %[[REF:.*]] = bmodelica.tensor_extract %[[ARR]][%[[INDEX0]], %[[INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[REF]]
        // CHECK-NEXT:     %[[CALL_RES:.*]] = bmodelica.call @f(%[[IDX0]], %[[IDX1]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CALL_RES]]
        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %0 = bmodelica.equation_template inductions = [%arg0, %arg1] {
            %4 = bmodelica.constant -1 : index
            %5 = bmodelica.add %arg0, %4 : (index, index) -> index
            %6 = bmodelica.add %arg1, %4 : (index, index) -> index
            %7 = bmodelica.variable_get @a : tensor<4x5xi32>
            %8 = bmodelica.tensor_extract %7[%5, %6] : tensor<4x5xi32>
            %9 = bmodelica.equation_side %8 : tuple<i32>
            %10 = bmodelica.call @f(%arg0, %arg1) : (index, index) -> i32
            %11 = bmodelica.equation_side %10 : tuple<i32>
            bmodelica.equation_sides %9, %11 : tuple<i32>, tuple<i32>
        }

        // COM: This template would trigger the CSE if the conflicting template were not ignored.
        %1 = bmodelica.equation_template inductions = [%arg0, %arg1] {
            %4 = bmodelica.constant -1 : index
            %5 = bmodelica.add %arg0, %4 : (index, index) -> index
            %6 = bmodelica.add %arg1, %4 : (index, index) -> index
            %7 = bmodelica.variable_get @b : tensor<4x5xi32>
            %8 = bmodelica.tensor_extract %7[%5, %6] : tensor<4x5xi32>
            %9 = bmodelica.equation_side %8 : tuple<i32>
            %10 = bmodelica.call @f(%arg0, %arg1) : (index, index) -> i32
            %11 = bmodelica.equation_side %10 : tuple<i32>
            bmodelica.equation_sides %9, %11 : tuple<i32>, tuple<i32>
        }

        // CHECK: bmodelica.dynamic
        bmodelica.dynamic {
            // CHECK-DAG: bmodelica.equation_instance %[[T0]] {indices = #modeling<multidim_range [1,4][1,5]>}
            // CHECK-DAG: bmodelica.equation_instance %[[T0]] {indices = #modeling<multidim_range [1,4][1,6]>}
            // CHECK-DAG: bmodelica.equation_instance %[[T1]] {indices = #modeling<multidim_range [1,4][1,5]>}
            bmodelica.equation_instance %0 {indices = #modeling<multidim_range [1,4][1,5]>} : !bmodelica.equation
            bmodelica.equation_instance %0 {indices = #modeling<multidim_range [1,4][1,6]>} : !bmodelica.equation
            bmodelica.equation_instance %1 {indices = #modeling<multidim_range [1,4][1,5]>} : !bmodelica.equation
        }
    }
}
