// RUN: modelica-opt %s --split-input-file --convert-bmodelica-raw-variables | FileCheck %s

// CHECK:       func.func @scalarVariableGet() -> f64 {
// CHECK-NEXT:      %[[alloca:.*]] = memref.alloc() : memref<f64>
// CHECK-NEXT:      %[[result:.*]] = memref.load %[[alloca]][] : memref<f64>
// CHECK-NEXT:      return %[[result]] : f64
// CHECK-NEXT:  }

func.func @scalarVariableGet() -> f64 {
    %0 = bmodelica.raw_variable {name = "x"} : memref<f64>
    %1 = bmodelica.raw_variable_get %0 : memref<f64> -> f64
    return %1 : f64
}

// -----

// CHECK:       func.func @scalarVariableSet(%[[arg0:.*]]: f64) {
// CHECK-NEXT:      %[[alloca:.*]] = memref.alloc() : memref<f64>
// CHECK-NEXT:      memref.store %[[arg0]], %[[alloca]][] : memref<f64>
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @scalarVariableSet(%arg0: f64) {
    %0 = bmodelica.raw_variable {name = "x"} : memref<f64>
    bmodelica.raw_variable_set %0, %arg0 : memref<f64>, f64
    return
}

// -----

// CHECK:       func.func @staticArrayVariableGet() -> memref<3x4xf64> {
// CHECK-NEXT:      %[[alloc:.*]] = memref.alloc() : memref<3x4xf64>
// CHECK-NEXT:      return %[[alloc]] : memref<3x4xf64>
// CHECK-NEXT:  }

func.func @staticArrayVariableGet() -> memref<3x4xf64> {
    %0 = bmodelica.raw_variable {name = "x"} : memref<3x4xf64>
    %1 = bmodelica.raw_variable_get %0 : memref<3x4xf64> -> memref<3x4xf64>
    return %1 : memref<3x4xf64>
}

// -----

// CHECK:       func.func @staticArrayVariableSet(%[[arg0:.*]]: memref<3x4xf64>) {
// CHECK-NEXT:      %[[alloc:.*]] = memref.alloc() : memref<3x4xf64>
// CHECK-NEXT:      memref.copy %[[arg0]], %[[alloc]] : memref<3x4xf64> to memref<3x4xf64>
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @staticArrayVariableSet(%arg0: memref<3x4xf64>) {
    %0 = bmodelica.raw_variable {name = "x"} : memref<3x4xf64>
    bmodelica.raw_variable_set %0, %arg0 : memref<3x4xf64>, memref<3x4xf64>
    return
}

// -----

// CHECK:       func.func @dynamicArrayVariableGet() -> memref<3x?xf64> {
// CHECK-NEXT:      %[[alloca:.*]] = memref.alloca() : memref<memref<3x?xf64>>
// CHECK-NEXT:      %[[alloc:.*]] = memref.alloc() : memref<3x0xf64>
// CHECK-NEXT:      %[[cast:.*]] = memref.cast %[[alloc]] : memref<3x0xf64> to memref<3x?xf64>
// CHECK-NEXT:      memref.store %[[cast]], %[[alloca]][] : memref<memref<3x?xf64>>
// CHECK-NEXT:      %[[result:.*]] = memref.load %[[alloca]][] : memref<memref<3x?xf64>>
// CHECK-NEXT:      return %[[result]] : memref<3x?xf64>
// CHECK-NEXT:  }

func.func @dynamicArrayVariableGet() -> memref<3x?xf64> {
    %0 = bmodelica.raw_variable {name = "x"} : memref<3x?xf64>
    %1 = bmodelica.raw_variable_get %0 : memref<3x?xf64> -> memref<3x?xf64>
    return %1 : memref<3x?xf64>
}

// -----

// CHECK:       func.func @dynamicArrayVariableSet(%[[arg0:.*]]: memref<3x?xf64>) {
// CHECK-NEXT:      %[[alloca:.*]] = memref.alloca() : memref<memref<3x?xf64>>
// CHECK-NEXT:      %[[alloc_1:.*]] = memref.alloc() : memref<3x0xf64>
// CHECK-NEXT:      %[[cast:.*]] = memref.cast %[[alloc_1]] : memref<3x0xf64> to memref<3x?xf64>
// CHECK-NEXT:      memref.store %[[cast]], %[[alloca]][] : memref<memref<3x?xf64>>
// CHECK-NEXT:      %[[one:.*]] = arith.constant 1 : index
// CHECK-NEXT:      %[[dim:.*]] = memref.dim %[[arg0]], %[[one]]
// CHECK-NEXT:      %[[alloc_2:.*]] = memref.alloc(%[[dim]]) : memref<3x?xf64>
// CHECK-NEXT:      memref.copy %[[arg0]], %[[alloc_2]]
// CHECK-NEXT:      %[[load:.*]] = memref.load %[[alloca]][] : memref<memref<3x?xf64>>
// CHECK-NEXT:      memref.dealloc %[[load]] : memref<3x?xf64>
// CHECK-NEXT:      memref.store %[[alloc_2]], %[[alloca]][] : memref<memref<3x?xf64>>
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @dynamicArrayVariableSet(%arg0: memref<3x?xf64>) {
    %0 = bmodelica.raw_variable {name = "x"} : memref<3x?xf64>
    bmodelica.raw_variable_set %0, %arg0 : memref<3x?xf64>, memref<3x?xf64>
    return
}
