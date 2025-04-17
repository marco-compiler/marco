// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Load from mono-dimensional array

// CHECK-LABEL: @Test
// CHECK-SAME: %{{.*}}: !bmodelica.array<2x!bmodelica.int>
// CHECK-SAME: %[[arg1:.*]]: index

func.func @Test(%arg0: !bmodelica.array<2x!bmodelica.int>, %arg1: index){

    // CHECK:       %[[conv:.*]] = bmodelica.array_to_tensor %arg0 : <2x!bmodelica.int> -> tensor<2x!bmodelica.int>
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: LoadOp out of bounds access"} {
    // CHECK-NEXT:      %[[constzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[indexzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[size:.*]] = bmodelica.size %[[conv]], %[[indexzero]] : (tensor<2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[gte:.*]] = bmodelica.gte %arg1, %[[constzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[lt:.*]] = bmodelica.lt %arg1, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.load %arg0[%arg1] : !bmodelica.array<2x!bmodelica.int>

    return
}

// -----

// COM: Load from multi-dimensional array

// CHECK-LABEL: @Test
// CHECK-SAME: %{{.*}}: !bmodelica.array<2x2x!bmodelica.int>
// CHECK-SAME: %[[arg:.*]]: index

func.func @Test(%arg0: !bmodelica.array<2x2x!bmodelica.int>, %arg1: index, %arg2: index){

    // CHECK:       %[[conv:.*]] = bmodelica.array_to_tensor %arg0 : <2x2x!bmodelica.int> -> tensor<2x2x!bmodelica.int>
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: LoadOp out of bounds access"} {
    // CHECK-NEXT:      %[[constzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[indexzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[size:.*]] = bmodelica.size %[[conv]], %[[indexzero]] : (tensor<2x2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[gte:.*]] = bmodelica.gte %arg1, %[[constzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[lt:.*]] = bmodelica.lt %arg1, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: LoadOp out of bounds access"} {
    // CHECK-NEXT:      %[[constzero_2:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[indexzero_2:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:      %[[size_2:.*]] = bmodelica.size %[[conv]], %[[indexzero_2]] : (tensor<2x2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[gte_2:.*]] = bmodelica.gte %arg2, %[[constzero_2]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[lt_2:.*]] = bmodelica.lt %arg2, %[[size_2]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond_2:.*]] = bmodelica.and %[[gte_2]], %[[lt_2]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond_2]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.load %arg0[%arg1, %arg2] : !bmodelica.array<2x2x!bmodelica.int>

    return
}