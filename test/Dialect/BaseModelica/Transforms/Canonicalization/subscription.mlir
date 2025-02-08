// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-SAME: (%[[arg:.*]]: !bmodelica.array<2x3x4x5x6x!bmodelica.int>)

func.func @test(%arg0: !bmodelica.array<2x3x4x5x6x!bmodelica.int>) -> (!bmodelica.array<6x!bmodelica.int>) {
    %0 = bmodelica.constant 0 : index
    %1 = bmodelica.constant 1 : index
    %2 = bmodelica.constant 2 : index
    %3 = bmodelica.constant 3 : index
    %5 = bmodelica.subscription %arg0[%0, %1] : !bmodelica.array<2x3x4x5x6x!bmodelica.int>, index, index -> !bmodelica.array<4x5x6x!bmodelica.int>
    %6 = bmodelica.subscription %5[%2, %3] : !bmodelica.array<4x5x6x!bmodelica.int>, index, index -> !bmodelica.array<6x!bmodelica.int>
    return %6 : !bmodelica.array<6x!bmodelica.int>

    // CHECK-DAG: %[[cst_0:.*]] = bmodelica.constant 0 : index
    // CHECK-DAG: %[[cst_1:.*]] = bmodelica.constant 1 : index
    // CHECK-DAG: %[[cst_2:.*]] = bmodelica.constant 2 : index
    // CHECK-DAG: %[[cst_3:.*]] = bmodelica.constant 3 : index
    // CHECK: %[[result:.*]] = bmodelica.subscription %[[arg]][%[[cst_0]], %[[cst_1]], %[[cst_2]], %[[cst_3]]]
    // CHECK: return %[[result]]
}
