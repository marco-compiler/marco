// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func --canonicalize | FileCheck %s

// Scalar variable.

// CHECK:       func.func @getter(%arg0: !llvm.ptr<i64>) -> f64 {
// CHECK-NEXT:      %[[cst:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      %[[result:.*]] = modelica.cast %[[cst]] : !modelica.real -> f64
// CHECK-NEXT:      return %[[result]] : f64
// CHECK-NEXT:  }

simulation.variable_getter @getter() -> f64 {
    %0 = modelica.constant #modelica.real<0.0>
    %1 = modelica.cast %0 : !modelica.real -> f64
    simulation.return %1 : f64
}

// -----

// Array variable.

// CHECK:       func.func @getter(%[[ptr:.*]]: !llvm.ptr<i64>) -> f64 {
// CHECK-DAG:       %[[array:.*]] = modelica.alloc : !modelica.array<2x3x4x!modelica.real>
// CHECK-DAG:       %[[i0_load:.*]] = llvm.load %[[ptr]]
// CHECK-DAG:       %[[i0:.*]] = arith.index_cast %[[i0_load]] : i64 to index
// CHECK-DAG:       %[[i1_addr:.*]] = llvm.getelementptr %[[ptr]][1]
// CHECK-DAG:       %[[i1_load:.*]] = llvm.load %[[i1_addr]]
// CHECK-DAG:       %[[i1:.*]] = arith.index_cast %[[i1_load]] : i64 to index
// CHECK-DAG:       %[[i2_addr:.*]] = llvm.getelementptr %[[ptr]][2]
// CHECK-DAG:       %[[i2_load:.*]] = llvm.load %[[i2_addr]]
// CHECK-DAG:       %[[i2:.*]] = arith.index_cast %[[i2_load]] : i64 to index
// CHECK:           %[[load:.*]] = modelica.load %[[array]][%[[i0]], %[[i1]], %[[i2]]]
// CHECK:           %[[result:.*]] = modelica.cast %[[load]] : !modelica.real -> f64
// CHECK:           return %[[result:.*]] : f64
// CHECK-NEXT:  }

simulation.variable_getter @getter(%arg0: index, %arg1: index, %arg2: index) -> f64 {
    %0 = modelica.alloc : !modelica.array<2x3x4x!modelica.real>
    %1 = modelica.load %0[%arg0, %arg1, %arg2] : !modelica.array<2x3x4x!modelica.real>
    %2 = modelica.cast %1 : !modelica.real -> f64
    simulation.return %2 : f64
}
