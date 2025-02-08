// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @staticDimension
// CHECK-SAME: %[[arg0:.*]]: !bmodelica.array<7x?x9x!bmodelica.int>

func.func @staticDimension(%arg0: !bmodelica.array<7x?x9x!bmodelica.int>) -> index {
    %0 = bmodelica.constant 2 : index
    %1 = bmodelica.dim %arg0, %0 : !bmodelica.array<7x?x9x!bmodelica.int>
    return %1 : index

    // CHECK-DAG: %[[cst_9:.*]] = bmodelica.constant 9 : index
    // CHECK: return %[[cst_9]]
}

// -----

// CHECK-LABEL: @dynamicDimension
// CHECK-SAME: %[[arg0:.*]]: !bmodelica.array<7x?x9x!bmodelica.int>

func.func @dynamicDimension(%arg0: !bmodelica.array<7x?x9x!bmodelica.int>) -> index {
    %0 = bmodelica.constant 1 : index
    %1 = bmodelica.dim %arg0, %0 : !bmodelica.array<7x?x9x!bmodelica.int>
    return %1 : index

    // CHECK-DAG: %[[cst_1:.*]] = bmodelica.constant 1 : index
    // CHECK-DAG: %[[dim:.*]] = bmodelica.dim %[[arg0]], %[[cst_1]]
    // CHECK: return %[[dim]]
}

// -----

// CHECK-LABEL: @unknownDimension
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<7x?x9x!bmodelica.int>, %[[arg1:.*]]: index)

func.func @unknownDimension(%arg0: !bmodelica.array<7x?x9x!bmodelica.int>, %arg1: index) -> index {
    %0 = bmodelica.dim %arg0, %arg1 : !bmodelica.array<7x?x9x!bmodelica.int>
    return %0 : index

    // CHECK-DAG: %[[dim:.*]] = bmodelica.dim %[[arg0]], %[[arg1]]
    // CHECK: return %[[dim]]
}
