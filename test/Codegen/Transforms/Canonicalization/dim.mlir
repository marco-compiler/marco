// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-SAME: %[[arg0:.*]]: !modelica.array<7x?x9x!modelica.int>
// CHECK-DAG: %[[cst_9:.*]] = modelica.constant 9 : index
// CHECK: return %[[cst_9]]

func.func @test(%arg0: !modelica.array<7x?x9x!modelica.int>) -> index {
    %0 = modelica.constant 2 : index
    %1 = modelica.dim %arg0, %0 : !modelica.array<7x?x9x!modelica.int>
    return %1 : index
}

// -----

// CHECK-LABEL: @test
// CHECK-SAME: %[[arg0:.*]]: !modelica.array<7x?x9x!modelica.int>
// CHECK-DAG: %[[cst_1:.*]] = modelica.constant 1 : index
// CHECK-DAG: %[[dim:.*]] = modelica.dim %[[arg0]], %[[cst_1]]
// CHECK: return %[[dim]]

func.func @test(%arg0: !modelica.array<7x?x9x!modelica.int>) -> index {
    %0 = modelica.constant 1 : index
    %1 = modelica.dim %arg0, %0 : !modelica.array<7x?x9x!modelica.int>
    return %1 : index
}

// -----

// CHECK-LABEL: @test
// CHECK-SAME: %[[arg0:.*]]: !modelica.array<7x?x9x!modelica.int>
// CHECK-SAME: %[[arg1:.*]]: index
// CHECK-DAG: %[[dim:.*]] = modelica.dim %[[arg0]], %[[arg1]]
// CHECK: return %[[dim]]

func.func @test(%arg0: !modelica.array<7x?x9x!modelica.int>, %arg1: index) -> index {
    %0 = modelica.dim %arg0, %arg1 : !modelica.array<7x?x9x!modelica.int>
    return %0 : index
}
