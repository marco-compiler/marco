// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s


// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @tensor : !bmodelica.variable<tensor<4x!bmodelica.real>>
    bmodelica.variable @var.x : !bmodelica.variable<tensor<2x3x!bmodelica.real>>

    %0 = bmodelica.constant #bmodelica.int_range<0, 2, 1> : !bmodelica<range index>
    %1 = bmodelica.constant 1 : index
    %2 = bmodelica.constant 0 : index
    %3 = bmodelica.variable_get @tensor : tensor<4x!bmodelica.real>
    %4 = bmodelica.variable_get @var.x : tensor<2x3x!bmodelica.real>

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: source array dimension greater than destination array dimension"} {
    // CHECK-NEXT:  %[[const_1:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:  %[[const_2:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[size_1:.*]] = bmodelica.size %3, %[[const_2]] : (tensor<4x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[size_2:.*]] = bmodelica.size %4, %[[const_1]] : (tensor<2x3x!bmodelica.real>, index) -> !bmodelica.int
    // CHECK-NEXT:  %[[lte:.*]] = bmodelica.lte %[[size_1]], %[[size_2]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:  bmodelica.yield %[[lte]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %5 = bmodelica.tensor_insert_slice %3, %4[%2, %0] : tensor<4x!bmodelica.real>, tensor<2x3x!bmodelica.real>, index, !bmodelica<range index> -> tensor<2x3x!bmodelica.real>

}