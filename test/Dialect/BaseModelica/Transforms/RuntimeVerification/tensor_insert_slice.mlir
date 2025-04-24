// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @rangeSubscript
// CHECK-SAME: (%[[arg0:.*]]: tensor<?xf64>, %[[arg1:.*]]: tensor<?xf64>, %[[arg2:.*]]: !bmodelica<range index>)

func.func @rangeSubscript(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: !bmodelica<range index>) -> tensor<?xf64> {
    // CHECK:       bmodelica.assert
    // CHECK-DAG:   %[[dimIndex:.*]] = arith.constant 0 : index
    // CHECK-DAG:   %[[dimSize:.*]] = tensor.dim %[[arg0]], %[[dimIndex]]
    // CHECK-DAG:   %[[rangeSize:.*]] = bmodelica.range_size %[[arg2]]
    // CHECK:       %[[condition:.*]] = bmodelica.eq %[[dimSize]], %[[rangeSize]]
    // CHECK:       bmodelica.yield %[[condition]]

    %1 = bmodelica.tensor_insert_slice %arg0, %arg1[%arg2] : tensor<?xf64>, tensor<?xf64>, !bmodelica<range index> -> tensor<?xf64>
    func.return %1 : tensor<?xf64>
}

// -----

// CHECK-LABEL: @implicitRangeSubscript
// CHECK-SAME: (%[[arg0:.*]]: tensor<?xf64>, %[[arg1:.*]]: tensor<?xf64>)

func.func @implicitRangeSubscript(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
    // CHECK: bmodelica.assert
    // CHECK: %[[sourceDimIndex:.*]] = arith.constant 0 : index
    // CHECK: %[[sourceDimSize:.*]] = tensor.dim %[[arg0]], %[[sourceDimIndex]]
    // CHECK: %[[destinationDimIndex:.*]] = arith.constant 0 : index
    // CHECK: %[[destinationDimSize:.*]] = tensor.dim %[[arg1]], %[[destinationDimIndex]]
    // CHECK: %[[condition:.*]] = bmodelica.eq %[[sourceDimSize]], %[[destinationDimSize]]
    // CHECK: bmodelica.yield %[[condition]]

    %1 = bmodelica.tensor_insert_slice %arg0, %arg1[] : tensor<?xf64>, tensor<?xf64> -> tensor<?xf64>
    func.return %1 : tensor<?xf64>
}

// -----

// CHECK-LABEL: @constantAndRangeSubscripts
// CHECK-SAME: (%[[arg0:.*]]: tensor<?xf64>, %[[arg1:.*]]: tensor<?x?xf64>, %[[arg2:.*]]: index, %[[arg3:.*]]: !bmodelica<range index>)

func.func @constantAndRangeSubscripts(%arg0: tensor<?xf64>, %arg1: tensor<?x?xf64>, %arg2: index, %arg3: !bmodelica<range index>) -> tensor<?x?xf64> {
    // CHECK:       bmodelica.assert
    // CHECK-DAG:   %[[dimIndex:.*]] = arith.constant 0 : index
    // CHECK-DAG:   %[[dimSize:.*]] = tensor.dim %[[arg0]], %[[dimIndex]]
    // CHECK-DAG:   %[[rangeSize:.*]] = bmodelica.range_size %[[arg3]]
    // CHECK:       %[[condition:.*]] = bmodelica.eq %[[dimSize]], %[[rangeSize]]
    // CHECK:       bmodelica.yield %[[condition]]

    %1 = bmodelica.tensor_insert_slice %arg0, %arg1[%arg2, %arg3] : tensor<?xf64>, tensor<?x?xf64>, index, !bmodelica<range index> -> tensor<?x?xf64>
    func.return %1 : tensor<?x?xf64>
}

// -----

// CHECK-LABEL: @constantAndImplicitRangeSubscripts
// CHECK-SAME: (%[[arg0:.*]]: tensor<?xf64>, %[[arg1:.*]]: tensor<?x?xf64>, %[[arg2:.*]]: index)

func.func @constantAndImplicitRangeSubscripts(%arg0: tensor<?xf64>, %arg1: tensor<?x?xf64>, %arg2: index) -> tensor<?x?xf64> {
    // CHECK:       bmodelica.assert
    // CHECK-DAG:   %[[sourceDimIndex:.*]] = arith.constant 0 : index
    // CHECK-DAG:   %[[sourceDimSize:.*]] = tensor.dim %[[arg0]], %[[sourceDimIndex]]
    // CHECK-DAG:   %[[destinationDimIndex:.*]] = arith.constant 1 : index
    // CHECK-DAG:   %[[destinationDimSize:.*]] = tensor.dim %[[arg1]], %[[destinationDimIndex]]
    // CHECK:       %[[condition:.*]] = bmodelica.eq %[[sourceDimSize]], %[[destinationDimSize]]
    // CHECK:       bmodelica.yield %[[condition]]

    %1 = bmodelica.tensor_insert_slice %arg0, %arg1[%arg2] : tensor<?xf64>, tensor<?x?xf64>, index -> tensor<?x?xf64>
    func.return %1 : tensor<?x?xf64>
}
