// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Store to mono-dimensional tensor

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @a : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @b.x : !bmodelica.variable<2x!bmodelica.real>

    %0 = bmodelica.constant 0 : index
    %1 = bmodelica.variable_get @a : !bmodelica.real
    %2 = bmodelica.variable_get @b.x : tensor<2x!bmodelica.real>

    // CHECK: bmodelica.assert {level = 2 : i64, message = "Model error: TensorInsertOp out of bounds access"} {
    // CHECK-NEXT:    %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:    %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:    %[[size:.*]] = bmodelica.size %2, %[[constzero]] : (tensor<2x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:    %[[gte:.*]] = bmodelica.gte %0, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    %[[lt:.*]]  = bmodelica.lt %0, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:    bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:    }
    
    %3 = bmodelica.tensor_insert %1, %2[%0] : !bmodelica.real, tensor<2x!bmodelica.real>, index -> tensor<2x!bmodelica.real>

}

// -----

// COM: Store to multi-dimensional tensor

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @a : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @b.x : !bmodelica.variable<2x2x!bmodelica.real>

    %0 = bmodelica.constant 0 : index
    %1 = bmodelica.constant 1 : index
    %2 = bmodelica.variable_get @a : !bmodelica.real
    %3 = bmodelica.variable_get @b.x : tensor<2x2x!bmodelica.real>

    // CHECK: bmodelica.assert {level = 2 : i64, message = "Model error: TensorInsertOp out of bounds access"} {
    // CHECK-NEXT:    %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:    %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:    %[[size:.*]] = bmodelica.size %3, %[[constzero]] : (tensor<2x2x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:    %[[gte:.*]] = bmodelica.gte %0, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    %[[lt:.*]]  = bmodelica.lt %0, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:    bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:    }

    // CHECK-NEXT:    bmodelica.assert {level = 2 : i64, message = "Model error: TensorInsertOp out of bounds access"} {
    // CHECK-NEXT:    %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:    %[[constone:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:    %[[size:.*]] = bmodelica.size %3, %[[constone]] : (tensor<2x2x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:    %[[gte:.*]] = bmodelica.gte %1, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    %[[lt:.*]]  = bmodelica.lt %1, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:    %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:    bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:    }
    
    %4 = bmodelica.tensor_insert %2, %3[%0, %1] : !bmodelica.real, tensor<2x2x!bmodelica.real>, index, index -> tensor<2x2x!bmodelica.real>

}