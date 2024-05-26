// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-linalg | FileCheck %s

// CHECK: @foo(%[[arg0:.*]]: tensor<3x4x5xf64>)
// CHECK: %[[destination:.*]] = tensor.empty() : tensor<3x4x5xf64>
// CHECK: %[[map:.*]] = linalg.map { bmodelica.tanh } ins(%[[arg0]] : tensor<3x4x5xf64>) outs(%[[destination]] : tensor<3x4x5xf64>)
// CHECK: return %[[map]]

func.func @foo(%arg0: tensor<3x4x5xf64>) -> (tensor<3x4x5xf64>) {
    %result = bmodelica.tanh %arg0 : tensor<3x4x5xf64> -> tensor<3x4x5xf64>
    return %result : tensor<3x4x5xf64>
}
