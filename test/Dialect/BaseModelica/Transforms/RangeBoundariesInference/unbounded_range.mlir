// RUN: modelica-opt %s --split-input-file --infer-range-boundaries | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[source:.*]]: tensor<6x5x4x3x2x!bmodelica.int>)

func.func @foo(%arg0: tensor<6x5x4x3x2x!bmodelica.int>) -> tensor<?x5x4x3x2x!bmodelica.int> {
    %0 = bmodelica.unbounded_range : !bmodelica<range index>
    %1 = bmodelica.tensor_view %arg0[%0] : tensor<6x5x4x3x2x!bmodelica.int>, !bmodelica<range index> -> tensor<?x5x4x3x2x!bmodelica.int>
    func.return %1 : tensor<?x5x4x3x2x!bmodelica.int>

    // CHECK: %[[range:.*]] = bmodelica.constant #bmodelica.int_range<0, 5, 1>
    // CHECK: %[[subscription:.*]] = bmodelica.tensor_view %[[source]][%[[range]]]
    // CHECK: return %[[subscription]]
}
