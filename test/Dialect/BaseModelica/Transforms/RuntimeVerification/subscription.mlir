// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Mono-dimensional

// CHECK-LABEL: @Test
// CHECK-SAME: %{{.*}}: !bmodelica.array<2x!bmodelica.int>
// CHECK-SAME: %[[arg1:.*]]: index

func.func @Test(%arg0: !bmodelica.array<2x!bmodelica.int>, %arg1: index) {

    // CHECK:       %[[conv:.*]] = bmodelica.array_to_tensor %arg0 : <2x!bmodelica.int> -> tensor<2x!bmodelica.int>
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: SubscriptionOp out of bounds access"} {
    // CHECK-NEXT:      %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[size:.*]] = bmodelica.size %[[conv]], %[[constzero]] : (tensor<2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[gte:.*]] = bmodelica.gte %arg1, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[lt:.*]] = bmodelica.lt %arg1, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.subscription %arg0[%arg1] : <2x!bmodelica.int>, index -> !bmodelica.array<!bmodelica.real>
    func.return
}

// -----

// COM: Multi-dimensional

// CHECK-LABEL: @Test
// CHECK-SAME: %{{.*}}: !bmodelica.array<2x2x!bmodelica.int>
// CHECK-SAME: %[[arg1:.*]]: index

func.func @Test(%arg0: !bmodelica.array<2x2x!bmodelica.int>, %arg1: index, %arg2: index) {

    // CHECK:       %[[conv:.*]] = bmodelica.array_to_tensor %arg0 : <2x2x!bmodelica.int> -> tensor<2x2x!bmodelica.int>
    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: SubscriptionOp out of bounds access"} {
    // CHECK-NEXT:      %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[constzero:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[size:.*]] = bmodelica.size %[[conv]], %[[constzero]] : (tensor<2x2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[gte:.*]] = bmodelica.gte %arg1, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[lt:.*]] = bmodelica.lt %arg1, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    // CHECK-NEXT:  bmodelica.assert {level = 2 : i64, message = "Model error: SubscriptionOp out of bounds access"} {
    // CHECK-NEXT:      %[[intzero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[constone:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:      %[[size:.*]] = bmodelica.size %[[conv]], %[[constone]] : (tensor<2x2x!bmodelica.int>, index) -> !bmodelica.int
    // CHECK-NEXT:      %[[gte:.*]] = bmodelica.gte %arg2, %[[intzero]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[lt:.*]] = bmodelica.lt %arg2, %[[size]] : (index, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      %[[cond:.*]] = bmodelica.and %[[gte]], %[[lt]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[cond]] : !bmodelica.bool
    // CHECK-NEXT:  }

    %0 = bmodelica.subscription %arg0[%arg1, %arg2] : <2x2x!bmodelica.int>, index, index -> !bmodelica.array<!bmodelica.int>
    func.return
}