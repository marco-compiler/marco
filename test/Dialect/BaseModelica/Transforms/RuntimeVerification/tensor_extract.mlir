// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// Mono-dimensional

// CHECK-LABEL: @Test
func.func @Test(%arg0: !bmodelica.array<3x!bmodelica.real>, %arg1: index){

    %0 = bmodelica.array_to_tensor %arg0 : <3x!bmodelica.real> -> tensor<3x!bmodelica.real>

    // CHECK: bmodelica.assert {level = 2 : i64, message = "Model error: TensorExtractOp out of bounds access"} {
    // CHECK-NEXT:  %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %0, %[[constzero]] : (tensor<3x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %arg1, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %arg1, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }
    
    %1 = bmodelica.tensor_extract %0[%arg1] : tensor<3x!bmodelica.real>

    return
}

// -----

// Multi-dimensional

// CHECK-LABEL: @Test
func.func @Test(%arg0: !bmodelica.array<3x3x!bmodelica.real>, %arg1: index, %arg2: index){
    %0 = bmodelica.array_to_tensor %arg0 : <3x3x!bmodelica.real> -> tensor<3x3x!bmodelica.real>

    // CHECK: bmodelica.assert {level = 2 : i64, message = "Model error: TensorExtractOp out of bounds access"} {
    // CHECK-NEXT:  %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %0, %[[constzero]] : (tensor<3x3x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %arg1, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %arg1, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: TensorExtractOp out of bounds access"} {
    // CHECK-NEXT:  %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[constone:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %0, %[[constone]] : (tensor<3x3x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %arg2, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %arg2, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }
    
    %1 = bmodelica.tensor_extract %0[%arg1, %arg2] : tensor<3x3x!bmodelica.real>

    return
}