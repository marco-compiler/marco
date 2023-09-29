// RUN: modelica-opt %s --split-input-file --infer-range-boundaries | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[source:.*]]: !modelica.array<6x5x4x3x2x!modelica.int>)
// CHECK-DAG: %[[zero:.*]] = modelica.constant 0
// CHECK-DAG: %[[one:.*]] = modelica.constant 1
// CHECK-DAG: %[[minus_one:.*]] = modelica.constant -1
// CHECK-DAG: %[[dimSize:.*]] = modelica.dim %[[source]], %[[zero]]
// CHECK-DAG: %[[upperBound:.*]] = modelica.add %[[dimSize]], %[[minus_one]]
// CHECK: %[[range:.*]] = modelica.range %[[zero]], %[[upperBound]], %[[one]]
// CHECK: %[[subscription:.*]] = modelica.subscription %[[source]][%[[range]]]
// CHECK: return %[[subscription]]

func.func @foo(%arg0: !modelica.array<6x5x4x3x2x!modelica.int>) -> !modelica.array<?x5x4x3x2x!modelica.int> {
    %0 = modelica.unbounded_range : !modelica<range index>
    %1 = modelica.subscription %arg0[%0] : !modelica.array<6x5x4x3x2x!modelica.int>, !modelica<range index> -> !modelica.array<?x5x4x3x2x!modelica.int>
    func.return %1 : !modelica.array<?x5x4x3x2x!modelica.int>
}
