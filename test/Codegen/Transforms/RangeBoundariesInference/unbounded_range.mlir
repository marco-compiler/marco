// RUN: modelica-opt %s --split-input-file --infer-range-boundaries | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[source:.*]]: !modelica.array<6x5x4x3x2x!modelica.int>)
// CHECK: %[[lowerBound:.*]] = modelica.constant 0
// CHECK: %[[dim:.*]] = modelica.constant 0 : index
// CHECK: %[[dimSize:.*]] = modelica.dim %[[source]], %[[dim]]
// CHECK: %[[offset:.*]] = modelica.constant -1
// CHECK: %[[upperBound:.*]] = modelica.add %[[dimSize]], %[[offset]]
// CHECK: %[[step:.*]] = modelica.constant 1
// CHECK: %[[range:.*]] = modelica.range %[[lowerBound]], %[[upperBound]], %[[step]]
// CHECK: %[[subscription:.*]] = modelica.subscription %[[source]][%[[range]]]
// CHECK: return %[[subscription]]

func.func @foo(%arg0: !modelica.array<6x5x4x3x2x!modelica.int>) -> !modelica.array<?x5x4x3x2x!modelica.int> {
    %0 = modelica.unbounded_range : !modelica<range index>
    %1 = modelica.subscription %arg0[%0] : !modelica.array<6x5x4x3x2x!modelica.int>, !modelica<range index> -> !modelica.array<?x5x4x3x2x!modelica.int>
    func.return %1 : !modelica.array<?x5x4x3x2x!modelica.int>
}
