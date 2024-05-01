// RUN: modelica-opt %s --split-input-file --infer-range-boundaries | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[source:.*]]: !bmodelica.array<6x5x4x3x2x!bmodelica.int>)
// CHECK-DAG: %[[zero:.*]] = bmodelica.constant 0
// CHECK-DAG: %[[one:.*]] = bmodelica.constant 1
// CHECK-DAG: %[[minus_one:.*]] = bmodelica.constant -1
// CHECK-DAG: %[[dimSize:.*]] = bmodelica.dim %[[source]], %[[zero]]
// CHECK-DAG: %[[upperBound:.*]] = bmodelica.add %[[dimSize]], %[[minus_one]]
// CHECK: %[[range:.*]] = bmodelica.range %[[zero]], %[[upperBound]], %[[one]]
// CHECK: %[[subscription:.*]] = bmodelica.subscription %[[source]][%[[range]]]
// CHECK: return %[[subscription]]

func.func @foo(%arg0: !bmodelica.array<6x5x4x3x2x!bmodelica.int>) -> !bmodelica.array<?x5x4x3x2x!bmodelica.int> {
    %0 = bmodelica.unbounded_range : !bmodelica<range index>
    %1 = bmodelica.subscription %arg0[%0] : !bmodelica.array<6x5x4x3x2x!bmodelica.int>, !bmodelica<range index> -> !bmodelica.array<?x5x4x3x2x!bmodelica.int>
    func.return %1 : !bmodelica.array<?x5x4x3x2x!bmodelica.int>
}
