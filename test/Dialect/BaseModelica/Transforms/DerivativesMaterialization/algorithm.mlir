// RUN: modelica-opt %s --split-input-file --derivatives-materialization --canonicalize | FileCheck %s

// CHECK-LABEL: @arrayVariable

bmodelica.model @arrayVariable {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>

    bmodelica.dynamic {
        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
            %1 = bmodelica.constant 3 : index
            %2 = bmodelica.tensor_extract %0[%1] : tensor<5x!bmodelica.real>
            %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
            bmodelica.variable_set @x[%1], %3 : index, !bmodelica.real
        }

        // CHECK:       bmodelica.algorithm {
        // CHECK-DAG:       %[[index:.*]] = bmodelica.constant 3 : index
        // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
        // CHECK-NEXT:      %[[extract:.*]] = bmodelica.tensor_extract %[[der_x]][%[[index]]]
        // CHECK-NEXT:      bmodelica.variable_set @x[%[[index]]], %[[extract]]
        // CHECK-NEXT:  }
    }
}

// -----

// COM: Array variable with not all indices being derived.
// COM: The remaining indices are intentionally considered as being derived, as per Modelica specification.

// CHECK-LABEL: @partialArrayVariable

bmodelica.model @partialArrayVariable {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>

    bmodelica.dynamic {
        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
            %1 = bmodelica.constant 3 : index
            %2 = bmodelica.tensor_extract %0[%1] : tensor<5x!bmodelica.real>
            %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
            bmodelica.variable_set @x[%1], %3 : index, !bmodelica.real
        }
    }

    // CHECK-NOT:   bmodelica.equation_template
}
