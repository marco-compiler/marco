// RUN: modelica-opt %s --split-input-file --convert-runtime-to-func --canonicalize | FileCheck %s

// Scalar variable.

// CHECK:       func.func @getter(%arg0: !llvm.ptr) -> f64 {
// CHECK-NEXT:      %[[cst:.*]] = bmodelica.constant 0.000000e+00 : f64
// CHECK-NEXT:      return %[[cst]] : f64
// CHECK-NEXT:  }

runtime.variable_getter @getter() -> f64 {
    %0 = bmodelica.constant #bmodelica<real 0.0>
    %1 = bmodelica.cast %0 : !bmodelica.real -> f64
    runtime.return %1 : f64
}

// -----

// Array variable.

// CHECK:       func.func @getter(%[[ptr:.*]]: !llvm.ptr) -> f64 {
// CHECK-DAG:       %[[array:.*]] = bmodelica.alloc : <2x3x4x!bmodelica.real>
// CHECK-DAG:       %[[i0_load:.*]] = llvm.load %[[ptr]]
// CHECK-DAG:       %[[i0:.*]] = arith.index_cast %[[i0_load]] : i64 to index
// CHECK-DAG:       %[[i1_addr:.*]] = llvm.getelementptr %[[ptr]][1]
// CHECK-DAG:       %[[i1_load:.*]] = llvm.load %[[i1_addr]]
// CHECK-DAG:       %[[i1:.*]] = arith.index_cast %[[i1_load]] : i64 to index
// CHECK-DAG:       %[[i2_addr:.*]] = llvm.getelementptr %[[ptr]][2]
// CHECK-DAG:       %[[i2_load:.*]] = llvm.load %[[i2_addr]]
// CHECK-DAG:       %[[i2:.*]] = arith.index_cast %[[i2_load]] : i64 to index
// CHECK:           %[[load:.*]] = bmodelica.load %[[array]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK:           %[[result:.*]] = bmodelica.cast %[[load]] : !bmodelica.real -> f64
// CHECK:           return %[[result:.*]] : f64
// CHECK-NEXT:  }

runtime.variable_getter @getter(%arg0: index, %arg1: index, %arg2: index) -> f64 {
    %0 = bmodelica.alloc : <2x3x4x!bmodelica.real>
    %1 = bmodelica.load %0[%arg0, %arg1, %arg2] : !bmodelica.array<2x3x4x!bmodelica.real>
    %2 = bmodelica.cast %1 : !bmodelica.real -> f64
    runtime.return %2 : f64
}
