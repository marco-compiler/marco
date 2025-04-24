// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<?x?xf64>, %[[arg1:.*]]: index, %[[arg2:.*]]: index)

func.func @Test(%arg0: !bmodelica.array<?x?xf64>, %arg1: index, %arg2: index) -> !bmodelica.array<f64> {
    // CHECK:  bmodelica.assert
    // CHECK:  %[[lowerBound:.*]] = bmodelica.constant 0 : index
    // CHECK:  %[[lbCondition:.*]] = bmodelica.gte %[[arg1]], %[[lowerBound]]
    // CHECK:  %[[dimIndex:.*]] = bmodelica.constant 0 : index
    // CHECK:  %[[dimSize:.*]] = bmodelica.dim %[[arg0]], %[[dimIndex]]
    // CHECK:  %[[ubCondition:.*]] = bmodelica.lt %[[arg1]], %[[dimSize]]
    // CHECK:  %[[condition:.*]] = bmodelica.and %[[lbCondition]], %[[ubCondition]]
    // CHECK:  bmodelica.yield %[[condition]]

    // CHECK:  bmodelica.assert
    // CHECK:  %[[lowerBound:.*]] = bmodelica.constant 0 : index
    // CHECK:  %[[lbCondition:.*]] = bmodelica.gte %[[arg2]], %[[lowerBound]]
    // CHECK:  %[[dimIndex:.*]] = bmodelica.constant 1 : index
    // CHECK:  %[[dimSize:.*]] = bmodelica.dim %[[arg0]], %[[dimIndex]]
    // CHECK:  %[[ubCondition:.*]] = bmodelica.lt %[[arg2]], %[[dimSize]]
    // CHECK:  %[[condition:.*]] = bmodelica.and %[[lbCondition]], %[[ubCondition]]
    // CHECK:  bmodelica.yield %[[condition]]

    %0 = bmodelica.subscription %arg0[%arg1, %arg2] : <?x?xf64>, index, index -> !bmodelica.array<f64>
    func.return %0 : !bmodelica.array<f64>
}
