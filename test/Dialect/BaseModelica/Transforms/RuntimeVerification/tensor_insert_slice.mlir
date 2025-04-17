// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: tensor<4x!bmodelica.real>, %[[arg1:.*]]: tensor<2x3x!bmodelica.real>, %[[arg2:.*]]: index)

func.func @Test(%arg0: tensor<4x!bmodelica.real>, %arg1: tensor<2x3x!bmodelica.real>, %arg2: index) -> tensor<2x3x!bmodelica.real> {
    %0 = bmodelica.constant #bmodelica.int_range<0, 2, 1> : !bmodelica<range index>

    // CHECK:       bmodelica.assert
    // CHECK-NEXT:  %[[dst_dim_idx:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:  %[[src_dim_idx:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[src_dim_size:.*]] = bmodelica.size %[[arg0]], %[[src_dim_idx]]
    // CHECK-NEXT:  %[[dst_dim_size:.*]] = bmodelica.size %[[arg1]], %[[dst_dim_idx]]
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.lte %[[src_dim_size]], %[[dst_dim_size]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %1 = bmodelica.tensor_insert_slice %arg0, %arg1[%arg2, %0] : tensor<4x!bmodelica.real>, tensor<2x3x!bmodelica.real>, index, !bmodelica<range index> -> tensor<2x3x!bmodelica.real>
    func.return %1 : tensor<2x3x!bmodelica.real>
}