// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// CHECK-LABEL: @Test

func.func @Test(%arg0: tensor<4x!bmodelica.real>,
                %arg1: tensor<2x3x!bmodelica.real>, %arg2: index){

    %0 = bmodelica.constant #bmodelica.int_range<0, 2, 1> : !bmodelica<range index>

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: source array dimension greater than destination array dimension"} {
    // CHECK-NEXT:      %[[const_1:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:      %[[const_2:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[size_1:.*]] = bmodelica.size %arg0, %[[const_2]] : (tensor<4x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[size_2:.*]] = bmodelica.size %arg1, %[[const_1]] : (tensor<2x3x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[lte:.*]] = bmodelica.lte %[[size_1]], %[[size_2]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[lte]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %1 = bmodelica.tensor_insert_slice %arg0, %arg1[%arg2, %0] : tensor<4x!bmodelica.real>, tensor<2x3x!bmodelica.real>, index, !bmodelica<range index> -> tensor<2x3x!bmodelica.real>

    return
}