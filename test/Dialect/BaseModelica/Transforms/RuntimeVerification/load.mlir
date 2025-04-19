// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Scalar load

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<!bmodelica.int>)

func.func @Test(%arg0: !bmodelica.array<!bmodelica.int>) -> !bmodelica.int {

    // CHECK-NOT: bmodelica.assert
    // CHECK: {{.*}} = bmodelica.load %[[arg0]][]

    %0 = bmodelica.load %arg0[] : !bmodelica.array<!bmodelica.int>
    func.return %0 : !bmodelica.int
}

// -----

// COM: Array load

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<2x?x!bmodelica.int>, %[[arg1:.*]]: index, %[[arg2:.*]]: index)

func.func @Test(%arg0: !bmodelica.array<2x?x!bmodelica.int>, %arg1: index, %arg2: index) -> !bmodelica.int {

    // CHECK:       %[[arg0_casted:.*]] = bmodelica.array_to_tensor %[[arg0]]

    // CHECK-NEXT:  bmodelica.assert
    // CHECK-NEXT:  %[[lower_bound:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[dim_idx:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:  %[[dim_size:.*]] = bmodelica.size %[[arg0_casted]], %[[dim_idx]]
    // CHECK-NEXT:  %[[subcond_1:.*]] = bmodelica.gte %[[arg1]], %[[lower_bound]]
    // CHECK-NEXT:  %[[subcond_2:.*]] = bmodelica.lt %[[arg1]], %[[dim_size]]
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[subcond_1]], %[[subcond_2]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    // CHECK:       bmodelica.assert
    // CHECK-NEXT:  %[[lower_bound:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:  %[[dim_idx:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:  %[[dim_size:.*]] = bmodelica.size %[[arg0_casted]], %[[dim_idx]]
    // CHECK-NEXT:  %[[subcond_1:.*]] = bmodelica.gte %[[arg2]], %[[lower_bound]]
    // CHECK-NEXT:  %[[subcond_2:.*]] = bmodelica.lt %[[arg2]], %[[dim_size]]
    // CHECK-NEXT:  %[[cond:.*]] = bmodelica.and %[[subcond_1]], %[[subcond_2]]
    // CHECK-NEXT:  bmodelica.yield %[[cond]] : !bmodelica.bool

    %0 = bmodelica.load %arg0[%arg1, %arg2] : !bmodelica.array<2x?x!bmodelica.int>
    func.return %0 : !bmodelica.int
}