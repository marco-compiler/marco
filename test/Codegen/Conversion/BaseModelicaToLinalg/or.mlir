// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-linalg | FileCheck %s

// CHECK: #[[map:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @staticTensors
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x4x5x6xi1>, %[[arg1:.*]]: tensor<3x4x5x6xi1>) -> tensor<3x4x5x6xi1>
// CHECK:   %[[destination:.*]] = tensor.empty()
// CHECK:   %[[result:.*]] = linalg.generic {indexing_maps = [#[[map]], #[[map]], #[[map]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[arg0]], %[[arg1]] : tensor<3x4x5x6xi1>, tensor<3x4x5x6xi1>) outs(%[[destination]] : tensor<3x4x5x6xi1>) {
// CHECK:   ^bb0(%[[in_0:.*]]: i1, %[[in_1:.*]]: i1, %{{.*}}: i1):
// CHECK:       %[[scalar_op:.*]] = bmodelica.or %[[in_0]], %[[in_1]]
// CHECK:       linalg.yield %[[scalar_op]]
// CHECK:   } -> tensor<3x4x5x6xi1>
// CHECK:   return %[[result]]

func.func @staticTensors(%arg0 : tensor<3x4x5x6xi1>, %arg1 : tensor<3x4x5x6xi1>) -> tensor<3x4x5x6xi1> {
    %0 = bmodelica.or %arg0, %arg1 : (tensor<3x4x5x6xi1>, tensor<3x4x5x6xi1>) -> tensor<3x4x5x6xi1>
    func.return %0 : tensor<3x4x5x6xi1>
}

// -----

// CHECK: #[[map:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @dynamicTensors
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x?x5xi1>, %[[arg1:.*]]: tensor<3x?x5xi1>) ->  tensor<3x?x5xi1>
// CHECK:   %[[one:.*]] = arith.constant 1 : index
// CHECK:   %[[dim:.*]] = tensor.dim %[[arg0]], %[[one]]
// CHECK:   %[[destination:.*]] = tensor.empty(%[[dim]])
// CHECK:   %[[result:.*]] = linalg.generic
// CHECK:   return %[[result]]

func.func @dynamicTensors(%arg0 : tensor<3x?x5xi1>, %arg1 : tensor<3x?x5xi1>) -> tensor<3x?x5xi1> {
    %0 = bmodelica.or %arg0, %arg1 : (tensor<3x?x5xi1>, tensor<3x?x5xi1>) -> tensor<3x?x5xi1>
    func.return %0 : tensor<3x?x5xi1>
}
