// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-tensor | FileCheck %s

// Static array.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<3x4xi64>) -> tensor<2xindex>
// CHECK:       %[[empty:.*]] = tensor.empty() : tensor<2xindex>
// CHECK:       %[[zero:.*]] = arith.constant 0 : index
// CHECK:       %[[dim:.*]] = tensor.dim %[[arg0]], %[[zero]]
// CHECK:       %[[insert:.*]] = tensor.insert %[[dim]] into %[[empty]][%[[zero]]]
// CHECK:       return %[[insert]]

func.func @foo(%arg0: tensor<3x4xi64>) -> tensor<2xindex> {
    %0 = bmodelica.size %arg0 : tensor<3x4xi64> -> tensor<2xindex>
    func.return %0 : tensor<2xindex>
}

// -----

// Dynamic array.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x?xi64>) -> tensor<2xindex>
// CHECK:       %[[empty:.*]] = tensor.empty() : tensor<2xindex>
// CHECK:       %[[zero:.*]] = arith.constant 0 : index
// CHECK:       %[[dim:.*]] = tensor.dim %[[arg0]], %[[zero]]
// CHECK:       %[[insert:.*]] = tensor.insert %[[dim]] into %[[empty]][%[[zero]]]
// CHECK:       return %[[insert]]

func.func @foo(%arg0: tensor<?x?xi64>) -> tensor<2xindex> {
    %0 = bmodelica.size %arg0 : tensor<?x?xi64> -> tensor<2xindex>
    func.return %0 : tensor<2xindex>
}

// -----

// Static array dimension.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<3x4xi64>, %[[arg1:.*]]: index) -> index
// CHECK: %[[result:.*]] = tensor.dim %[[arg0]], %[[arg1]]
// CHECK: return %[[result]]

func.func @foo(%arg0: tensor<3x4xi64>, %arg1: index) -> index {
    %0 = bmodelica.size %arg0, %arg1 : (tensor<3x4xi64>, index) -> index
    func.return %0 : index
}

// -----

// Dynamic array dimension.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: tensor<?x?xi64>, %[[arg1:.*]]: index) -> index
// CHECK: %[[result:.*]] = tensor.dim %[[arg0]], %[[arg1]]
// CHECK: return %[[result]]

func.func @foo(%arg0: tensor<?x?xi64>, %arg1: index) -> index {
    %0 = bmodelica.size %arg0, %arg1 : (tensor<?x?xi64>, index) -> index
    func.return %0 : index
}
