// RUN: modelica-opt %s --split-input-file --one-shot-bufferize | FileCheck %s

// CHECK-LABEL: @unusedScalarVariable
// CHECK-SAME:  ()
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @unusedScalarVariable() {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<f64>
    func.return
}

// -----

// CHECK-LABEL: @scalarVariableGet
// CHECK-SAME:  () -> f64
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x"} : memref<f64>
// CHECK-NEXT:      %[[result:.*]] = bmodelica.raw_variable_get %[[variable]] : memref<f64> -> f64
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

func.func @scalarVariableGet() -> f64 {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<f64>
    %1 = bmodelica.raw_variable_get %0 : tensor<f64> -> f64
    func.return %1 : f64
}

// -----

// CHECK-LABEL: @scalarVariableSet
// CHECK-SAME:  (%[[arg0:.*]]: f64)
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x"} : memref<f64>
// CHECK-NEXT:      bmodelica.raw_variable_set %[[variable]], %[[arg0]] : memref<f64>, f64
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @scalarVariableSet(%arg0: f64) {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<f64>
    bmodelica.raw_variable_set %0, %arg0 : tensor<f64>, f64
    func.return
}

// -----

// CHECK-LABEL: @scalarVariableSetAndGet
// CHECK-SAME:  (%[[arg0:.*]]: f64) -> f64
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x"} : memref<f64>
// CHECK-NEXT:      bmodelica.raw_variable_set %[[variable]], %[[arg0]] : memref<f64>, f64
// CHECK-NEXT:      %[[result:.*]] = bmodelica.raw_variable_get %[[variable]] : memref<f64> -> f64
// CHECK-NEXT:      return %[[result]] : f64
// CHECK-NEXT:  }

func.func @scalarVariableSetAndGet(%arg0: f64) -> f64 {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<f64>
    bmodelica.raw_variable_set %0, %arg0 : tensor<f64>, f64
    %1 = bmodelica.raw_variable_get %0 : tensor<f64> -> f64
    func.return %1 : f64
}

// -----

// CHECK-LABEL: @unusedStaticArrayVariable
// CHECK-SAME:  ()
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @unusedStaticArrayVariable() {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<3x4xf64>
    func.return
}

// -----

// CHECK-LABEL: @staticArrayVariableGet
// CHECK-SAME:  () -> tensor<3x4xf64>
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x"} : memref<3x4xf64>
// CHECK-NEXT:      %[[get:.*]] = bmodelica.raw_variable_get %[[variable]] : memref<3x4xf64> -> memref<3x4xf64>
// CHECK-NEXT:      %[[result:.*]] = bufferization.to_tensor %[[get]] : memref<3x4xf64>
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

func.func @staticArrayVariableGet() -> tensor<3x4xf64> {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<3x4xf64>
    %1 = bmodelica.raw_variable_get %0 : tensor<3x4xf64> -> tensor<3x4xf64>
    func.return %1 : tensor<3x4xf64>
}

// -----

// CHECK-LABEL: @staticArrayVariableSet
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x4xf64>)
// CHECK-DAG:       %[[value:.*]] = bufferization.to_memref %[[arg0]]
// CHECK-DAG:       %[[variable:.*]] = bmodelica.raw_variable {name = "x"} : memref<3x4xf64>
// CHECK:           bmodelica.raw_variable_set %[[variable]], %[[value]]
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @staticArrayVariableSet(%arg0: tensor<3x4xf64>) {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<3x4xf64>
    bmodelica.raw_variable_set %0, %arg0 : tensor<3x4xf64>, tensor<3x4xf64>
    func.return
}

// -----

// CHECK-LABEL: @staticArrayVariableSetAndGet
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x4xf64>) -> tensor<3x4xf64>
// CHECK-DAG:       %[[arg0_to_memref:.*]] = bufferization.to_memref %[[arg0]] : tensor<3x4xf64> to memref<3x4xf64, strided<[?, ?], offset: ?>>
// CHECK-DAG:       %[[variable:.*]] = bmodelica.raw_variable {name = "x"} : memref<3x4xf64>
// CHECK-NEXT:      bmodelica.raw_variable_set %[[variable]], %[[arg0_to_memref]] : memref<3x4xf64>, memref<3x4xf64, strided<[?, ?], offset: ?>>
// CHECK-NEXT:      %[[result:.*]] = bmodelica.raw_variable_get %[[variable]] : memref<3x4xf64> -> memref<3x4xf64>
// CHECK-NEXT:      %[[result_to_tensor:.*]] = bufferization.to_tensor %[[result]] : memref<3x4xf64> to tensor<3x4xf64>
// CHECK-NEXT:      return %[[result_to_tensor]] : tensor<3x4xf64>
// CHECK-NEXT:  }

func.func @staticArrayVariableSetAndGet(%arg0: tensor<3x4xf64>) -> tensor<3x4xf64> {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<3x4xf64>
    bmodelica.raw_variable_set %0, %arg0 : tensor<3x4xf64>, tensor<3x4xf64>
    %1 = bmodelica.raw_variable_get %0 : tensor<3x4xf64> -> tensor<3x4xf64>
    func.return %1 : tensor<3x4xf64>
}

// -----

// CHECK-LABEL: @unusedDynamicArrayVariable
// CHECK-SAME:  ()
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @unusedDynamicArrayVariable() {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<3x?xf64>
    func.return
}

// -----

// CHECK-LABEL: @dynamicArrayVariableGet
// CHECK-SAME:  () -> tensor<3x?xf64>
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x"} : memref<3x?xf64>
// CHECK-NEXT:      %[[get:.*]] = bmodelica.raw_variable_get %[[variable]] : memref<3x?xf64> -> memref<3x?xf64>
// CHECK-NEXT:      %[[result:.*]] = bufferization.to_tensor %[[get]] : memref<3x?xf64>
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

func.func @dynamicArrayVariableGet() -> tensor<3x?xf64> {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<3x?xf64>
    %1 = bmodelica.raw_variable_get %0 : tensor<3x?xf64> -> tensor<3x?xf64>
    func.return %1 : tensor<3x?xf64>
}

// -----

// CHECK-LABEL: @dynamicArrayVariableSet
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x?xf64>)
// CHECK-DAG:       %[[value:.*]] = bufferization.to_memref %[[arg0]]
// CHECK-DAG:       %[[variable:.*]] = bmodelica.raw_variable {name = "x"} : memref<3x?xf64>
// CHECK:           bmodelica.raw_variable_set %[[variable]], %[[value]]
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @dynamicArrayVariableSet(%arg0: tensor<3x?xf64>) {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<3x?xf64>
    bmodelica.raw_variable_set %0, %arg0 : tensor<3x?xf64>, tensor<3x?xf64>
    func.return
}

// -----

// CHECK-LABEL: @dynamicArrayVariableSetAndGet
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x?xf64>) -> tensor<3x?xf64>
// CHECK-DAG:       %[[arg0_to_memref:.*]] = bufferization.to_memref %[[arg0]] : tensor<3x?xf64> to memref<3x?xf64, strided<[?, ?], offset: ?>>
// CHECK-DAG:       %[[variable:.*]] = bmodelica.raw_variable {name = "x"} : memref<3x?xf64>
// CHECK-NEXT:      bmodelica.raw_variable_set %[[variable]], %[[arg0_to_memref]] : memref<3x?xf64>, memref<3x?xf64, strided<[?, ?], offset: ?>>
// CHECK-NEXT:      %[[result:.*]] = bmodelica.raw_variable_get %[[variable]] : memref<3x?xf64> -> memref<3x?xf64>
// CHECK-NEXT:      %[[result_to_tensor:.*]] = bufferization.to_tensor %[[result]] : memref<3x?xf64> to tensor<3x?xf64>
// CHECK-NEXT:      return %[[result_to_tensor]] : tensor<3x?xf64>
// CHECK-NEXT:  }

func.func @dynamicArrayVariableSetAndGet(%arg0: tensor<3x?xf64>) -> tensor<3x?xf64> {
    %0 = bmodelica.raw_variable {name = "x"} : tensor<3x?xf64>
    bmodelica.raw_variable_set %0, %arg0 : tensor<3x?xf64>, tensor<3x?xf64>
    %1 = bmodelica.raw_variable_get %0 : tensor<3x?xf64> -> tensor<3x?xf64>
    func.return %1 : tensor<3x?xf64>
}
