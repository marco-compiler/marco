// RUN: modelica-opt %s --split-input-file --call-cse | FileCheck %s

// CHECK-LABEL: @SubsetRepresentative
module @SubsetRepresentative {
    bmodelica.function @f {
        bmodelica.variable @x : !bmodelica.variable<i32, input>
        bmodelica.variable @y : !bmodelica.variable<i32, output>
    }

    // CHECK-LABEL: @M
    bmodelica.model @M  {
        // CHECK-NEXT: bmodelica.variable @[[CSE:.*]] : !bmodelica.variable<5xi32>
        // CHECK-NEXT: bmodelica.variable @a
        // CHECK-NEXT: bmodelica.variable @b
        bmodelica.variable @a : !bmodelica.variable<4x5xi32>
        bmodelica.variable @b : !bmodelica.variable<4x5xi32>

        // CHECK:      %[[T0:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @a
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]]]
        // CHECK-NEXT:     %[[ADD:.*]] = bmodelica.add %[[CSE_REF]], %[[IDX0]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[ADD]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %0 = bmodelica.equation_template inductions = [%arg0, %arg1] {
            %4 = bmodelica.constant -1 : index
            %5 = bmodelica.add %arg0, %4 : (index, index) -> index
            %6 = bmodelica.add %arg1, %4 : (index, index) -> index
            %7 = bmodelica.variable_get @a : tensor<4x5xi32>
            %8 = bmodelica.tensor_extract %7[%5, %6] : tensor<4x5xi32>
            %9 = bmodelica.equation_side %8 : tuple<i32>

            %10 = bmodelica.call @f(%arg1) : (index) -> i32
            %11 = bmodelica.add %10, %arg0: (i32, index) -> index
            %12 = bmodelica.equation_side %11 : tuple<index>

            bmodelica.equation_sides %9, %12 : tuple<i32>, tuple<index>
        }

        // CHECK:      %[[T0:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @b
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]]]
        // CHECK-NEXT:     %[[ADD:.*]] = bmodelica.add %[[CSE_REF]], %[[IDX0]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[ADD]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %1 = bmodelica.equation_template inductions = [%arg0, %arg1] {
            %4 = bmodelica.constant -1 : index
            %5 = bmodelica.add %arg0, %4 : (index, index) -> index
            %6 = bmodelica.add %arg1, %4 : (index, index) -> index
            %7 = bmodelica.variable_get @b : tensor<4x5xi32>
            %8 = bmodelica.tensor_extract %7[%5, %6] : tensor<4x5xi32>
            %9 = bmodelica.equation_side %8 : tuple<i32>

            %10 = bmodelica.call @f(%arg1) : (index) -> i32
            %11 = bmodelica.add %10, %arg0: (i32, index) -> index
            %12 = bmodelica.equation_side %11 : tuple<index>

            bmodelica.equation_sides %9, %12 : tuple<i32>, tuple<index>
        }

        bmodelica.dynamic {
            bmodelica.equation_instance %0 {indices = #modeling<multidim_range [1,4][1,5]>} : !bmodelica.equation
            bmodelica.equation_instance %1 {indices = #modeling<multidim_range [1,4][1,5]>} : !bmodelica.equation
        }

        // CHECK:      %[[T_CSE:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]]]
        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[OFFSET:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     %[[CALL_RES:.*]] = bmodelica.call @f(%[[IDX0]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CALL_RES]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]

        // CHECK:      bmodelica.dynamic
        // CHECK-NEXT:     bmodelica.equation_instance %[[T_CSE]] {indices = #modeling<multidim_range [1,5]>}
    }
}

// -----

// CHECK-LABEL: @InvertedEqualInductionRanges
module @InvertedEqualInductionRanges {
    bmodelica.function @f {
        bmodelica.variable @x : !bmodelica.variable<i32, input>
        bmodelica.variable @y : !bmodelica.variable<i32, input>
        bmodelica.variable @z : !bmodelica.variable<i32, output>
    }

    // CHECK-LABEL: @M
    bmodelica.model @M  {
        // CHECK-NEXT: bmodelica.variable @[[CSE:.*]] : !bmodelica.variable<5x5xi32>
        // CHECK-NEXT: bmodelica.variable @a
        // CHECK-NEXT: bmodelica.variable @b
        bmodelica.variable @a : !bmodelica.variable<5x5xi32>
        bmodelica.variable @b : !bmodelica.variable<5x5xi32>

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
            %7 = bmodelica.variable_get @a : tensor<5x5xi32>
            %8 = bmodelica.tensor_extract %7[%5, %6] : tensor<5x5xi32>
            %9 = bmodelica.equation_side %8 : tuple<i32>

            %10 = bmodelica.call @f(%arg0, %arg1) : (index, index) -> i32
            %11 = bmodelica.equation_side %10 : tuple<i32>

            bmodelica.equation_sides %9, %11 : tuple<i32>, tuple<i32>
        }

        // CHECK:      %[[T1:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @b
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[CSE_OFFSET0:.*]] = bmodelica.constant -1
        // COM: Note that the cse indices are reversed compared to T0
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[CSE_OFFSET0]]
        // CHECK-NEXT:     %[[CSE_OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[CSE_OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX1]], %[[CSE_INDEX0]]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %1 = bmodelica.equation_template inductions = [%arg0, %arg1] {
            %4 = bmodelica.constant -1 : index
            %5 = bmodelica.add %arg0, %4 : (index, index) -> index
            %6 = bmodelica.add %arg1, %4 : (index, index) -> index
            %7 = bmodelica.variable_get @b : tensor<5x5xi32>
            %8 = bmodelica.tensor_extract %7[%5, %6] : tensor<5x5xi32>
            %9 = bmodelica.equation_side %8 : tuple<i32>

            %10 = bmodelica.call @f(%arg1, %arg0) : (index, index) -> i32
            %11 = bmodelica.equation_side %10 : tuple<i32>

            bmodelica.equation_sides %9, %11 : tuple<i32>, tuple<i32>
        }

        bmodelica.dynamic {
            bmodelica.equation_instance %0 {indices = #modeling<multidim_range [1,5][1,5]>} : !bmodelica.equation
            bmodelica.equation_instance %1 {indices = #modeling<multidim_range [1,5][1,5]>} : !bmodelica.equation
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
        // CHECK-NEXT:     bmodelica.equation_instance %[[T_CSE]] {indices = #modeling<multidim_range [1,5][1,5]>}
    }
}

// -----

// CHECK-LABEL: @InductionRangeOffset
module @InductionRangeOffset {
    bmodelica.function @f {
        bmodelica.variable @x : !bmodelica.variable<i32, input>
        bmodelica.variable @y : !bmodelica.variable<i32, input>
        bmodelica.variable @z : !bmodelica.variable<i32, output>
    }

    // CHECK-LABEL: @M
    bmodelica.model @M  {
        // CHECK-NEXT: bmodelica.variable @[[CSE:.*]] : !bmodelica.variable<5x6xi32>
        // CHECK-NEXT: bmodelica.variable @a
        // CHECK-NEXT: bmodelica.variable @b
        bmodelica.variable @a : !bmodelica.variable<5x6xi32>
        bmodelica.variable @b : !bmodelica.variable<6x5xi32>

        // CHECK:      %[[T0:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant 2
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[OFFSET1:.*]] = bmodelica.constant -3
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET1]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @a
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[CSE_OFFSET0:.*]] = bmodelica.constant 2
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[CSE_OFFSET0]]
        // CHECK-NEXT:     %[[CSE_OFFSET1:.*]] = bmodelica.constant -3
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[CSE_OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]], %[[CSE_INDEX1]]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %0 = bmodelica.equation_template inductions = [%arg0, %arg1] {
            %offset0 = bmodelica.constant 2 : index
            %index0 = bmodelica.add %arg0, %offset0 : (index, index) -> index
            %offset1 = bmodelica.constant -3 : index
            %index1 = bmodelica.add %arg1, %offset1 : (index, index) -> index
            %7 = bmodelica.variable_get @a : tensor<5x6xi32>
            %8 = bmodelica.tensor_extract %7[%index0, %index1] : tensor<5x6xi32>
            %9 = bmodelica.equation_side %8 : tuple<i32>

            %10 = bmodelica.call @f(%arg0, %arg1) : (index, index) -> i32
            %11 = bmodelica.equation_side %10 : tuple<i32>

            bmodelica.equation_sides %9, %11 : tuple<i32>, tuple<i32>
        }

        // CHECK:      %[[T1:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -3
        // CHECK-NEXT:     %[[VAR_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[OFFSET1:.*]] = bmodelica.constant 2
        // CHECK-NEXT:     %[[VAR_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET1]]
        // CHECK-NEXT:     %[[VAR_ARR:.*]] = bmodelica.variable_get @b
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX0]], %[[VAR_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[CSE_OFFSET0:.*]] = bmodelica.constant 2
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX1]], %[[CSE_OFFSET0]]
        // CHECK-NEXT:     %[[CSE_OFFSET1:.*]] = bmodelica.constant -3
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX0]], %[[CSE_OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]], %[[CSE_INDEX1]]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %1 = bmodelica.equation_template inductions = [%arg0, %arg1] {
            %offset0 = bmodelica.constant -3 : index
            %index0 = bmodelica.add %arg0, %offset0 : (index, index) -> index
            %offset1 = bmodelica.constant 2 : index
            %index1 = bmodelica.add %arg1, %offset1 : (index, index) -> index
            %7 = bmodelica.variable_get @b : tensor<6x5xi32>
            %8 = bmodelica.tensor_extract %7[%index0, %index1] : tensor<6x5xi32>
            %9 = bmodelica.equation_side %8 : tuple<i32>

            %10 = bmodelica.call @f(%arg1, %arg0) : (index, index) -> i32
            %11 = bmodelica.equation_side %10 : tuple<i32>

            bmodelica.equation_sides %9, %11 : tuple<i32>, tuple<i32>
        }

        bmodelica.dynamic {
            bmodelica.equation_instance %0 {indices = #modeling<multidim_range [-2,2][3,8]>} : !bmodelica.equation
            bmodelica.equation_instance %1 {indices = #modeling<multidim_range [3,8][-2,2]>} : !bmodelica.equation
        }

        // CHECK:      %[[T_CSE:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]], %[[IDX1:.*]]]
        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant 2
        // CHECK-NEXT:     %[[CSE_INDEX0:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK-NEXT:     %[[OFFSET1:.*]] = bmodelica.constant -3
        // CHECK-NEXT:     %[[CSE_INDEX1:.*]] = bmodelica.add %[[IDX1]], %[[OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX0]], %[[CSE_INDEX1]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     %[[CALL_RES:.*]] = bmodelica.call @f(%[[IDX0]], %[[IDX1]])
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CALL_RES]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]

        // CHECK:      bmodelica.dynamic
        // CHECK-NEXT:     bmodelica.equation_instance %[[T_CSE]] {indices = #modeling<multidim_range [-2,2][3,8]>}
    }
}
