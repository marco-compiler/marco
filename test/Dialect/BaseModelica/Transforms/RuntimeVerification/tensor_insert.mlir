// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x?xf64>, %[[arg1:.*]]: index, %[[arg2:.*]]: index, %[[arg3:.*]]: f64)

func.func @Test(%arg0: tensor<?x?xf64>, %arg1: index, %arg2: index, %arg3: f64) -> tensor<?x?xf64> {
    // CHECK:  bmodelica.assert
    // CHECK:  %[[lowerBound:.*]] = bmodelica.constant 0 : index
    // CHECK:  %[[lbCondition:.*]] = bmodelica.gte %[[arg1]], %[[lowerBound]]
    // CHECK:  %[[dimIndex:.*]] = arith.constant 0 : index
    // CHECK:  %[[dimSize:.*]] = tensor.dim %[[arg0]], %[[dimIndex]]
    // CHECK:  %[[ubCondition:.*]] = bmodelica.lt %[[arg1]], %[[dimSize]]
    // CHECK:  %[[condition:.*]] = bmodelica.and %[[lbCondition]], %[[ubCondition]]
    // CHECK:  bmodelica.yield %[[condition]]

    // CHECK:  bmodelica.assert
    // CHECK:  %[[lowerBound:.*]] = bmodelica.constant 0 : index
    // CHECK:  %[[lbCondition:.*]] = bmodelica.gte %[[arg2]], %[[lowerBound]]
    // CHECK:  %[[dimIndex:.*]] = arith.constant 1 : index
    // CHECK:  %[[dimSize:.*]] = tensor.dim %[[arg0]], %[[dimIndex]]
    // CHECK:  %[[ubCondition:.*]] = bmodelica.lt %[[arg2]], %[[dimSize]]
    // CHECK:  %[[condition:.*]] = bmodelica.and %[[lbCondition]], %[[ubCondition]]
    // CHECK:  bmodelica.yield %[[condition]]
    
    %1 = bmodelica.tensor_insert %arg3, %arg0[%arg1, %arg2] : f64, tensor<?x?xf64>, index, index -> tensor<?x?xf64>
    func.return %1 : tensor<?x?xf64>
}
