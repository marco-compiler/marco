// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Integer values

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: !bmodelica.array<2x!bmodelica.int>, %[[arg1:.*]]: !bmodelica.int)

func.func @Test(%arg0: !bmodelica.array<2x!bmodelica.int>, %arg1: !bmodelica.int) -> !bmodelica.array<2x!bmodelica.int> {

    // CHECK:           bmodelica.assert 
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      %[[cond:.*]] = memref.alloc() : memref<!bmodelica.bool>
    // CHECK-NEXT:      %[[true:.*]] = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
    // CHECK-NEXT:      memref.store %[[true]], %[[cond]][] : memref<!bmodelica.bool>
    // CHECK-NEXT:      %[[neq:.*]] = bmodelica.neq %[[arg1]], %[[zero]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:      memref.store %[[neq]], %[[cond]][] : memref<!bmodelica.bool>
    // CHECK-NEXT:      %[[all:.*]] = memref.load %[[cond]][] : memref<!bmodelica.bool>
    // CHECK-NEXT:      bmodelica.yield %[[all]] : !bmodelica.bool

    %0 = bmodelica.div_ew %arg0, %arg1 : (!bmodelica.array<2x!bmodelica.int>, !bmodelica.int) -> !bmodelica.array<2x!bmodelica.int>
    func.return %0 : !bmodelica.array<2x!bmodelica.int>
}

// -----

// COM: Real values

// CHECK-LABEL: @Test
// CHECK-SAME: (%{{.*}}: !bmodelica.array<2x!bmodelica.real>, %[[arg1:.*]]: !bmodelica.real)

func.func @Test(%arg0: !bmodelica.array<2x!bmodelica.real>, %arg1: !bmodelica.real) -> !bmodelica.array<2x!bmodelica.real> {

    // CHECK:           bmodelica.assert
    // CHECK-NEXT:      %[[const:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-NEXT:      %[[cond:.*]] = memref.alloc() : memref<!bmodelica.bool>
    // CHECK-NEXT:      %[[true:.*]] = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
    // CHECK-NEXT:      memref.store %[[true]], %[[cond]][] : memref<!bmodelica.bool>
    // CHECK-NEXT:      %[[const_1:.*]] = bmodelica.constant #bmodelica<real 1.000000e-04> : !bmodelica.real
    // CHECK-NEXT:      %[[abs:.*]] = bmodelica.abs %[[arg1]] : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT:      %[[gte:.*]] = bmodelica.gte %[[abs]], %[[const_1]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:      memref.store %[[gte]], %[[cond]][] : memref<!bmodelica.bool>
    // CHECK-NEXT:      %[[all:.*]] = memref.load %[[cond]][] : memref<!bmodelica.bool>
    // CHECK-NEXT:      bmodelica.yield %[[all]] : !bmodelica.bool

    %0 = bmodelica.div_ew %arg0, %arg1 : (!bmodelica.array<2x!bmodelica.real>, !bmodelica.real) -> !bmodelica.array<2x!bmodelica.real>
    func.return %0 : !bmodelica.array<2x!bmodelica.real>
}

// -----

// COM: Array values

// CHECK-LABEL: @Test
// CHECK-SAME: (%[[arg0:.*]]: tensor<2x!bmodelica.real>, %[[arg1:.*]]: tensor<2x!bmodelica.real>)

func.func @Test(%arg0: tensor<2x!bmodelica.real>, %arg1: tensor<2x!bmodelica.real>) -> tensor<2x!bmodelica.real> {

    // CHECK:           bmodelica.assert
    // CHECK-NEXT:      %[[const_0:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-NEXT:      %[[cond:.*]] = memref.alloc() : memref<!bmodelica.bool>
    // CHECK-NEXT:      %[[true:.*]] = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
    // CHECK-NEXT:      memref.store %[[true]], %[[cond]][] : memref<!bmodelica.bool>
    // CHECK-NEXT:      %[[const_0:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[const_1:.*]] = bmodelica.constant 1 : index
    // CHECK-NEXT:      %[[const_0_2:.*]] = bmodelica.constant 0 : index
    // CHECK-NEXT:      %[[size:.*]] = bmodelica.size %arg1, %[[const_0_2]] : (tensor<2x!bmodelica.real>, index) -> index
    // CHECK-NEXT:      scf.for %[[idx:.*]] = %[[const_0]] to %[[size]] step %[[const_1]]
    // CHECK-NEXT:      %[[val:.*]] = bmodelica.tensor_extract %arg1[%[[idx]]] : tensor<2x!bmodelica.real>
    // CHECK-NEXT:      %[[const_eps:.*]] = bmodelica.constant #bmodelica<real 1.000000e-04> : !bmodelica.real
    // CHECK-NEXT:      %[[abs:.*]] = bmodelica.abs %[[val]] : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT:      %[[gte:.*]] = bmodelica.gte %[[abs]], %[[const_eps]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:      %[[oldRes:.*]] = memref.load %[[cond]][] : memref<!bmodelica.bool>
    // CHECK-NEXT:      %[[res:.*]] = bmodelica.and %[[oldRes]], %[[gte]] : (!bmodelica.bool, !bmodelica.bool) -> !bmodelica.bool
    // CHECK-NEXT:      memref.store %[[res]], %[[cond]][] : memref<!bmodelica.bool>
    // CHECK:           %[[all:.*]] = memref.load %[[cond]][] : memref<!bmodelica.bool>
    // CHECK-NEXT:      bmodelica.yield %[[all]] : !bmodelica.bool

    %0 = bmodelica.div_ew %arg0, %arg1 : (tensor<2x!bmodelica.real>, tensor<2x!bmodelica.real>) -> tensor<2x!bmodelica.real>
    func.return %0 : tensor<2x!bmodelica.real>
}