// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// Mono-dimensional

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @a : !bmodelica.variable<!bmodelica.array<3x!bmodelica.real>>
    bmodelica.variable @b : !bmodelica.variable<!bmodelica.array<!bmodelica.real>>

    %0 = bmodelica.constant 0 : index
    %1 = bmodelica.variable_get @b : !bmodelica.array<!bmodelica.real>
    %2 = bmodelica.variable_get @a : !bmodelica.array<3x!bmodelica.real>
    %3 = bmodelica.array_to_tensor %2 : <3x!bmodelica.real> -> tensor<3x!bmodelica.real>

    // CHECK: bmodelica.assert {level = 2 : i64, message = "Model error: TensorExtractOp out of bounds access"} {
    // CHECK-NEXT:  %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %3, %[[constzero]] : (tensor<3x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %0, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %0, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }
    
    %4 = bmodelica.tensor_extract %3[%0] : tensor<3x!bmodelica.real>

}

// -----

// Multi-dimensional

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @a : !bmodelica.variable<!bmodelica.array<3x3x!bmodelica.real>>
    bmodelica.variable @b : !bmodelica.variable<!bmodelica.array<!bmodelica.real>>

    %0 = bmodelica.constant 0 : index
    %1 = bmodelica.constant 1 : index
    %2 = bmodelica.variable_get @b : !bmodelica.array<!bmodelica.real>
    %3 = bmodelica.variable_get @a : !bmodelica.array<3x3x!bmodelica.real>
    %4 = bmodelica.array_to_tensor %3 : <3x3x!bmodelica.real> -> tensor<3x3x!bmodelica.real>

    // CHECK: bmodelica.assert {level = 2 : i64, message = "Model error: TensorExtractOp out of bounds access"} {
    // CHECK-NEXT:  %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %4, %[[constzero]] : (tensor<3x3x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %0, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %0, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: TensorExtractOp out of bounds access"} {
    // CHECK-NEXT:  %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[constone:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:  %[[size:.*]] = bmodelica.size %4, %[[constone]] : (tensor<3x3x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[gte:.*]] = bmodelica.gte %1, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[lt:.*]] = bmodelica.lt %1, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }
    
    %5 = bmodelica.tensor_extract %4[%0, %1] : tensor<3x3x!bmodelica.real>

}