// RUN: modelica-opt %s --call-cse | FileCheck %s

// CHECK-LABEL: @Test
module @Test {
    bmodelica.function @f {
        bmodelica.variable @x : !bmodelica.variable<i32, input>
        bmodelica.variable @y : !bmodelica.variable<i32, output>

        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : i32
            bmodelica.variable_set @y, %0 : i32
        }
    }

    // CHECK-LABEL: @Equations
    bmodelica.model @Equations {
        // CHECK-NEXT: bmodelica.variable @[[CSE:.*]] : !bmodelica.variable<4xi32>
        // CHECK-NEXT: bmodelica.variable @a
        // CHECK-NEXT: bmodelica.variable @b
        bmodelica.variable @a : !bmodelica.variable<4xi32>
        bmodelica.variable @b : !bmodelica.variable<4xi32>

        // CHECK:      %[[T0:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK:          %[[VAR_ARR:.*]] = bmodelica.variable_get @a
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX]]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %t0 = bmodelica.equation_template inductions = [%i] {
            %0 = bmodelica.constant -1 : index
            %1 = bmodelica.add %i, %0 : (index, index) -> index
            %2 = bmodelica.variable_get @a : tensor<4xi32>
            %3 = bmodelica.tensor_extract %2[%1] : tensor<4xi32>
            %4 = bmodelica.equation_side %3 : tuple<i32>

            %5 = bmodelica.call @f(%i) : (index) -> i32
            %6 = bmodelica.equation_side %5 : tuple<i32>

            bmodelica.equation_sides %4, %6 : tuple<i32>, tuple<i32>
        }

        // CHECK:      %[[T1:.*]] = bmodelica.equation_template inductions = [%[[IDX0:.*]]]
        // CHECK-NEXT:     %[[OFFSET0:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[VAR_INDEX:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET0]]
        // CHECK:          %[[VAR_ARR:.*]] = bmodelica.variable_get @b
        // CHECK-NEXT:     %[[VAR_REF:.*]] = bmodelica.tensor_extract %[[VAR_ARR]][%[[VAR_INDEX]]]
        // CHECK-NEXT:     %[[LHS:.*]] = bmodelica.equation_side %[[VAR_REF]]

        // CHECK-NEXT:     %[[CSE_ARR:.*]] = bmodelica.variable_get @[[CSE]]
        // CHECK-NEXT:     %[[OFFSET1:.*]] = bmodelica.constant -1
        // CHECK-NEXT:     %[[CSE_INDEX:.*]] = bmodelica.add %[[IDX0]], %[[OFFSET1]]
        // CHECK-NEXT:     %[[CSE_REF:.*]] = bmodelica.tensor_extract %[[CSE_ARR]][%[[CSE_INDEX]]]
        // CHECK-NEXT:     %[[RHS:.*]] = bmodelica.equation_side %[[CSE_REF]]

        // CHECK-NEXT:     bmodelica.equation_sides %[[LHS]], %[[RHS]]
        %t1 = bmodelica.equation_template inductions = [%i] {
            %0 = bmodelica.constant -1 : index
            %1 = bmodelica.add %i, %0 : (index, index) -> index
            %2 = bmodelica.variable_get @b : tensor<4xi32>
            %3 = bmodelica.tensor_extract %2[%1] : tensor<4xi32>
            %4 = bmodelica.equation_side %3 : tuple<i32>

            %5 = bmodelica.call @f(%i) : (index) -> i32
            %6 = bmodelica.equation_side %5 : tuple<i32>

            bmodelica.equation_sides %4, %6 : tuple<i32>, tuple<i32>
        }


        bmodelica.dynamic {
            bmodelica.equation_instance %t0, indices = {[1,4]}
            bmodelica.equation_instance %t1, indices = {[1,4]}
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
        // CHECK-NEXT:     bmodelica.equation_instance %[[T_CSE]], indices = {[1,4]}
    }
}