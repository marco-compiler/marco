// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x?x!bmodelica.int>, %[[arg1:.*]]: index, %[[arg2:.*]]: index)

func.func @Test(%arg0: tensor<2x?x!bmodelica.int>, %arg1: index, %arg2: index) -> tensor<2x?x!bmodelica.int> {
    %0 = bmodelica.constant #bmodelica<int 8> : !bmodelica.int

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: TensorInsertOp out of bounds access"} {
    // CHECK-NEXT:      %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[size:.*]] = bmodelica.size %[[arg0]], %[[constzero]] : (tensor<2x?x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[gte:.*]] = bmodelica.gte %[[arg1]], %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[lt:.*]]  = bmodelica.lt %[[arg1]], %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: TensorInsertOp out of bounds access"} {
    // CHECK-NEXT:      %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[constone:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:      %[[size:.*]] = bmodelica.size %[[arg0]], %[[constone]] : (tensor<2x?x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[gte:.*]] = bmodelica.gte %[[arg2]], %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[lt:.*]]  = bmodelica.lt %[[arg2]], %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }
    
    %1 = bmodelica.tensor_insert %0, %arg0[%arg1, %arg2] : !bmodelica.int, tensor<2x?x!bmodelica.int>, index, index -> tensor<2x?x!bmodelica.int>
    func.return %1 : tensor<2x?x!bmodelica.int>
}