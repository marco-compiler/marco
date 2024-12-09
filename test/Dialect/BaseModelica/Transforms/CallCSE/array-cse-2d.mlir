// RUN: modelica-opt %s --call-cse | FileCheck %s

// CHECK-LABEL: @Test
module @Test {
    bmodelica.function @f {
        bmodelica.variable @x : !bmodelica.variable<i32, input>
        bmodelica.variable @y : !bmodelica.variable<i32, input>
        bmodelica.variable @z : !bmodelica.variable<i32, output>
        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : i32
            %1 = bmodelica.variable_get @y : i32
            %2 = bmodelica.add %0, %1 : (i32, i32) -> i32
            bmodelica.variable_set @z, %2 : i32
        }
    }

    // CHECK-LABEL: @Equations2d
    bmodelica.model @Equations2d  {
        // CHECK-NEXT: bmodelica.variable @[[CSE:.*]] : !bmodelica.variable<4x5xi32>
        // CHECK-NEXT: bmodelica.variable @a
        // CHECK-NEXT: bmodelica.variable @b
        // CHECK-NEXT: bmodelica.variable @c
        // CHECK-NEXT: bmodelica.variable @d
        // CHECK-NEXT: bmodelica.variable @e
        bmodelica.variable @a : !bmodelica.variable<4x5xi32>
        bmodelica.variable @b : !bmodelica.variable<4x5xi32>
        bmodelica.variable @c : !bmodelica.variable<5x4xi32>
        bmodelica.variable @d : !bmodelica.variable<4x5x6xi32>
        bmodelica.variable @e : !bmodelica.variable<4x6x5xi32>

        // COM: Standard 2d pt. 1

        // CHECK:      %[[T0:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @a
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[CSE_OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[CSE_OFFSET0]]
        // CHECK-NEXT:     %[[CSE_OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[CSE_OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]], %[[CSE_INDEX1]]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

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

        // COM: Standard 2d pt. 2

        // CHECK:      %[[T1:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @b
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[CSE_OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[CSE_OFFSET0]]
        // CHECK-NEXT:     %[[CSE_OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[CSE_OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]], %[[CSE_INDEX1]]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
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

        // COM: Reversed induction ranges

        // CHECK:      %[[T2:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @c
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[CSE_OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX1]], %[[CSE_OFFSET0]]
        // CHECK-NEXT:     %[[CSE_OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX0]], %[[CSE_OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]], %[[CSE_INDEX1]]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %2 = bmodelica.equation_template inductions = [%arg0, %arg1] {
            %4 = bmodelica.constant -1 : index
            %5 = bmodelica.add %arg0, %4 : (index, index) -> index
            %6 = bmodelica.add %arg1, %4 : (index, index) -> index
            %7 = bmodelica.variable_get @c : tensor<5x4xi32>
            %8 = bmodelica.tensor_extract %7[%5, %6] : tensor<5x4xi32>
            %9 = bmodelica.equation_side %8 : tuple<i32>

            %10 = bmodelica.call @f(%arg1, %arg0) : (index, index) -> i32
            %11 = bmodelica.equation_side %10 : tuple<i32>

            bmodelica.equation_sides %9, %11 : tuple<i32>, tuple<i32>
        }

        // COM: Only use a subset of inductions

        // CHECK:      %[[T3:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]], %[[IDX2:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX2:.*]] = bmodelica.add %[[IDX2]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @d
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]], %[[VAR_INDEX2]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[CSE_OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[CSE_OFFSET0]]
        // CHECK-NEXT:     %[[CSE_OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[CSE_OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]], %[[CSE_INDEX1]]]
        // CHECK-NEXT:     %[[ADDITION:.*]] = bmodelica.add %[[CSE_REF]], %[[IDX2]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[ADDITION]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %3 = bmodelica.equation_template inductions = [%arg0, %arg1, %arg2] {
            %4 = bmodelica.constant -1 : index
            %5 = bmodelica.add %arg0, %4 : (index, index) -> index
            %6 = bmodelica.add %arg1, %4 : (index, index) -> index
            %7 = bmodelica.add %arg2, %4 : (index, index) -> index
            %8 = bmodelica.variable_get @d : tensor<4x5x6xi32>
            %9 = bmodelica.tensor_extract %8[%5, %6, %7] : tensor<4x5x6xi32>
            %10 = bmodelica.equation_side %9 : tuple<i32>

            %11 = bmodelica.call @f(%arg0, %arg1) : (index, index) -> i32
            %12 = bmodelica.add %11, %arg2 : (i32, index) -> index
            %13 = bmodelica.equation_side %12 : tuple<index>

            bmodelica.equation_sides %10, %13 : tuple<i32>, tuple<index>
        }

        // COM: Same as %3 but with different induction permutation

        // CHECK:      %[[T4:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]], %[[IDX2:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX2:.*]] = bmodelica.add %[[IDX2]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @e
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]], %[[VAR_INDEX2]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[CSE_OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[CSE_OFFSET0]]
        // CHECK-NEXT:     %[[CSE_OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX2]], %[[CSE_OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]], %[[CSE_INDEX1]]]
        // CHECK-NEXT:     %[[ADDITION:.*]] = bmodelica.add %[[CSE_REF]], %[[IDX1]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[ADDITION]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %4 = bmodelica.equation_template inductions = [%arg0, %arg1, %arg2] {
            %4 = bmodelica.constant -1 : index
            %5 = bmodelica.add %arg0, %4 : (index, index) -> index
            %6 = bmodelica.add %arg1, %4 : (index, index) -> index
            %7 = bmodelica.add %arg2, %4 : (index, index) -> index
            %8 = bmodelica.variable_get @e : tensor<4x6x5xi32>
            %9 = bmodelica.tensor_extract %8[%5, %6, %7] : tensor<4x6x5xi32>
            %10 = bmodelica.equation_side %9 : tuple<i32>

            %11 = bmodelica.call @f(%arg0, %arg2) : (index, index) -> i32
            %12 = bmodelica.add %11, %arg1 : (i32, index) -> index
            %13 = bmodelica.equation_side %12 : tuple<index>

            bmodelica.equation_sides %10, %13 : tuple<i32>, tuple<index>
        }

        bmodelica.dynamic {
            bmodelica.equation_instance %0 {indices = #modeling<multidim_range [1,4][1,5]>} : !bmodelica.equation
            bmodelica.equation_instance %1 {indices = #modeling<multidim_range [1,4][1,5]>} : !bmodelica.equation
            bmodelica.equation_instance %2 {indices = #modeling<multidim_range [1,5][1,4]>} : !bmodelica.equation
            bmodelica.equation_instance %3 {indices = #modeling<multidim_range [1,4][1,5][1,6]>} : !bmodelica.equation
            bmodelica.equation_instance %4 {indices = #modeling<multidim_range [1,4][1,6][1,5]>} : !bmodelica.equation
        }

        // CHECK:      %[[T_CSE:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[CSE_OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[CSE_OFFSET0]]
        // CHECK-NEXT:     %[[CSE_OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[CSE_OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]], %[[CSE_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     %[[CALL_RES:.*]] = bmodelica.call @f(%[[IDX0]], %[[IDX1]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CALL_RES]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]

        // CHECK:      bmodelica.dynamic
        // CHECK-NEXT:     bmodelica.equation_instance %[[T_CSE]] {indices = #modeling<multidim_range [1,4][1,5]>}
    }
}