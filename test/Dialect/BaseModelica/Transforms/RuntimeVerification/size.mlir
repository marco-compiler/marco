// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: One-dimensional tensor

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: tensor<3x!bmodelica.real>)

func.func @Test(%arg0: tensor<3x!bmodelica.real>) -> index {

    // CHECK:       %[[index:.*]] = bmodelica.constant
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[ndims:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:      %[[cond1:.*]] = bmodelica.gte %[[index]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond2:.*]] = bmodelica.lt %[[index]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.constant 1 : index
    %1 = bmodelica.size %arg0, %0 : (tensor<3x!bmodelica.real>, index) -> index
    func.return %1 : index
}

// -----

// COM: One-dimensional tensor, dimension index as function argument

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: tensor<3x!bmodelica.real>, %[[arg1:.*]]: index)

func.func @Test(%arg0: tensor<3x!bmodelica.real>, %arg1: index) -> index {

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[ndims:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:      %[[cond1:.*]] = bmodelica.gte %[[arg1]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.size %arg0, %arg1 : (tensor<3x!bmodelica.real>, index) -> index
    func.return %0 : index
}

// -----

// COM: One-dimensional tensor, without dimension of interest

// CHECK-LABEL: @Test
// CHECK-SAME: %[[arg0:.*]]: tensor<3x!bmodelica.real>

func.func @Test(%arg0: tensor<3x!bmodelica.real>) -> tensor<1xindex> {

    // CHECK: %{{.*}} = bmodelica.size %[[arg0]] : tensor<3x!bmodelica.real> -> tensor<1xindex>

    %0 = bmodelica.size %arg0 : tensor<3x!bmodelica.real> -> tensor<1xindex>
    func.return %0 : tensor<1xindex>
}

// -----

// COM: Multi-dimensional dynamic tensor

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: tensor<2x?x3x!bmodelica.real>)

func.func @Test(%arg0: tensor<2x?x3x!bmodelica.real>) -> index {

    // CHECK:       %[[index:.*]] = bmodelica.constant
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:      %[[cond1:.*]] = bmodelica.gte %[[index]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond2:.*]] = bmodelica.lt %[[index]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.constant 1 : index
    %1 = bmodelica.size %arg0, %0 : (tensor<2x?x3x!bmodelica.real>, index) -> index
    func.return %1 : index
}

// -----

// COM: Multi-dimensional dynamic tensor, dimension index as function argument

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: tensor<2x?x3x!bmodelica.real>, %[[arg1:.*]]: index)

func.func @Test(%arg0: tensor<2x?x3x!bmodelica.real>, %arg1: index) -> index {

    // CHECK:       bmodelica.assert {level = 2 : i64, message = "Model error: dimension index out of bounds"} {
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[ndims:.*]] = bmodelica.constant 3 : index
    // CHECK-NEXT:      %[[cond1:.*]] = bmodelica.gte %[[arg1]], %[[zero]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond2:.*]] = bmodelica.lt %[[arg1]], %[[ndims]] : (index, index) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[cond1]], %[[cond2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.size %arg0, %arg1 : (tensor<2x?x3x!bmodelica.real>, index) -> index
    func.return %0 : index
}

// -----

// COM: Multi-dimensional dynamic tensor, without dimension of interest

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x?x3x!bmodelica.real>)

func.func @Test(%arg0: tensor<2x?x3x!bmodelica.real>) -> index {

    // CHECK: %{{.*}} = bmodelica.size %[[arg0]] : tensor<2x?x3x!bmodelica.real> -> index

    %0 = bmodelica.size %arg0 : tensor<2x?x3x!bmodelica.real> -> index
    func.return %0 : index
}
