// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @constant
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x!bmodelica.real>, %[[arg1:.*]]: index)

func.func @constant(%arg0: tensor<?x!bmodelica.real>, %arg1: index) -> !bmodelica.real {
    // CHECK:  bmodelica.assert
    // CHECK:  %[[lowerBound:.*]] = bmodelica.constant 0 : index
    // CHECK:  %[[lbCondition:.*]] = bmodelica.gte %[[arg1]], %[[lowerBound]]
    // CHECK:  %[[dimIndex:.*]] = arith.constant 0 : index
    // CHECK:  %[[dimSize:.*]] = tensor.dim %[[arg0]], %[[dimIndex]]
    // CHECK:  %[[ubCondition:.*]] = bmodelica.lt %[[arg1]], %[[dimSize]]
    // CHECK:  %[[condition:.*]] = bmodelica.and %[[lbCondition]], %[[ubCondition]]
    // CHECK:  bmodelica.yield %[[condition]]

    %0 = bmodelica.tensor_view %arg0[%arg1] : tensor<?x!bmodelica.real>, index -> tensor<!bmodelica.real>
    %1 = bmodelica.tensor_extract %0[] : tensor<!bmodelica.real>
    func.return %1 : !bmodelica.real
}

// -----

// CHECK-LABEL: @range
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x!bmodelica.real>, %[[arg1:.*]]: !bmodelica<range index>)

func.func @range(%arg0: tensor<?x!bmodelica.real>, %arg1: !bmodelica<range index>) -> tensor<?x!bmodelica.real> {
    // CHECK:       bmodelica.assert
    // CHECK-DAG:   %[[rangeSize:.*]] = bmodelica.range_size %[[arg1]]
    // CHECK-DAG:   %[[dimIndex:.*]] = arith.constant 0 : index
    // CHECK-DAG:   %[[dimSize:.*]] = tensor.dim %[[arg0]], %[[dimIndex]]
    // CHECK:       %[[condition:.*]] = bmodelica.lte %[[rangeSize]], %[[dimSize]]
    // CHECK:       bmodelica.yield %[[condition]]

    %0 = bmodelica.tensor_view %arg0[%arg1] : tensor<?x!bmodelica.real>, !bmodelica<range index> -> tensor<?x!bmodelica.real>
    func.return %0 : tensor<?x!bmodelica.real>
}
