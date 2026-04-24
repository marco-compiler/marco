// RUN: modelica-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @Scalar

bmodelica.function @Scalar {
    bmodelica.variable @var : !bmodelica.variable<!bmodelica.real>

    bmodelica.algorithm {
        %0 = bmodelica.constant #bmodelica<real 0.0> : !bmodelica.real
        bmodelica.variable.set @var, %0 : !bmodelica.real

        // CHECK: %[[value:.*]] = bmodelica.constant
        // CHECK: bmodelica.variable.set @var, %[[value]] : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @Array

bmodelica.function @Array {
    bmodelica.variable @var : !bmodelica.variable<2x3x!bmodelica.real>

    bmodelica.algorithm {
        %0 = tensor.empty() : tensor<2x3x!bmodelica.real>
        bmodelica.variable.set @var, %0 : tensor<2x3x!bmodelica.real>

        // CHECK: %[[value:.*]] = tensor.empty
        // CHECK: bmodelica.variable.set @var, %[[value]] : tensor<2x3x!bmodelica.real>
    }
}

// -----

// CHECK-LABEL: @ArrayElement

bmodelica.function @ArrayElement {
    bmodelica.variable @var : !bmodelica.variable<2x3x!bmodelica.real>

    bmodelica.algorithm {
        %0 = bmodelica.constant #bmodelica<real 0.0> : !bmodelica.real
        %1 = arith.constant 0 : index
        %2 = arith.constant 1 : index
        bmodelica.variable.set @var[%1, %2 : index, index], %0 : !bmodelica.real

        // CHECK-DAG: %[[value:.*]] = bmodelica.constant
        // CHECK-DAG: %[[idx0:.*]] = arith.constant 0
        // CHECK-DAG: %[[idx1:.*]] = arith.constant 1
        // CHECK: bmodelica.variable.set @var[%[[idx0]], %[[idx1]] : index, index], %[[value]] : !bmodelica.real
    }
}
