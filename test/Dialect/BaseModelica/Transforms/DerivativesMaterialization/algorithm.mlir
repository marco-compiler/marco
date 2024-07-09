// RUN: modelica-opt %s --split-input-file --derivatives-materialization --canonicalize | FileCheck %s

// Check variable declaration and derivatives map.

// CHECK-LABEL: @Test
// CHECK-SAME: der = [<@x, @der_x, {[0,4]}>]
// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
// CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<5x!bmodelica.real>

bmodelica.model @Test {
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
}

// -----

// Check variable usage.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.algorithm {
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[extract:.*]] = bmodelica.tensor_extract %[[der_x]][%[[index]]]
// CHECK-NEXT:      bmodelica.variable_set @x[%[[index]]], %[[extract]]
// CHECK-NEXT:  }

bmodelica.model @Test {
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
}

// -----

// Check start value.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.start @der_x {
// CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[tensor:.*]] = bmodelica.tensor_broadcast %[[zero]] : !bmodelica.real -> tensor<5x!bmodelica.real>
// CHECK-NEXT:      bmodelica.yield %[[tensor]]
// CHECK-NEXT:  }

bmodelica.model @Test {
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
}

// -----

// Check equations for non-derived indices.

// CHECK-NOT:   bmodelica.equation_template

bmodelica.model @Test {
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
}
