// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-SAME: %[[arg:.*]]: !modelica.array<2x3x4x5x6x!modelica.int>
// CHECK-DAG: %[[cst_0:.*]] = modelica.constant 0 : index
// CHECK-DAG: %[[cst_1:.*]] = modelica.constant 1 : index
// CHECK-DAG: %[[cst_2:.*]] = modelica.constant 2 : index
// CHECK-DAG: %[[cst_3:.*]] = modelica.constant 3 : index
// CHECK-DAG: %[[cst_4:.*]] = modelica.constant 4 : index
// CHECK: %[[result:.*]] = modelica.load %[[arg]][%[[cst_0]], %[[cst_1]], %[[cst_2]], %[[cst_3]], %[[cst_4]]]
// CHECK: return %[[result]]

func.func @test(%arg0: !modelica.array<2x3x4x5x6x!modelica.int>) -> (!modelica.int) {
    %0 = modelica.constant 0 : index
    %1 = modelica.constant 1 : index
    %2 = modelica.constant 2 : index
    %3 = modelica.constant 3 : index
    %4 = modelica.constant 4 : index
    %5 = modelica.subscription %arg0[%0, %1] : !modelica.array<2x3x4x5x6x!modelica.int>, index, index -> !modelica.array<4x5x6x!modelica.int>
    %6 = modelica.subscription %5[%2, %3] : !modelica.array<4x5x6x!modelica.int>, index, index -> !modelica.array<6x!modelica.int>
    %7 = modelica.load %6[%4] : !modelica.array<6x!modelica.int>
    return %7 : !modelica.int
}
