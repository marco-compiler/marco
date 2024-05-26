// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-func | FileCheck %s

// CHECK:       func.func @scalarArgument(%[[arg0:.*]]: f64) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @scalarArgument(%arg0: f64) {
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @staticTensorArgument(%[[arg0:.*]]: tensor<5x3xf64>) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @staticTensorArgument(%arg0: tensor<5x3xf64>) {
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @dynamicTensorArgument(%[[arg0:.*]]: tensor<5x?xf64>) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @dynamicTensorArgument(%arg0: tensor<5x?xf64>) {
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @scalarResult() -> f64 {
// CHECK-NEXT:      %[[result:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

bmodelica.raw_function @scalarResult() -> f64 {
    %0 = arith.constant 0.0 : f64
    bmodelica.raw_return %0 : f64
}

// -----

// CHECK:       func.func @staticTensorResult() -> tensor<3x5xf64> {
// CHECK-NEXT:      %[[x:.*]] = tensor.empty() : tensor<3x5xf64>
// CHECK-NEXT:      return %[[x]]
// CHECK-NEXT:  }

bmodelica.raw_function @staticTensorResult() -> tensor<3x5xf64> {
    %0 = tensor.empty() : tensor<3x5xf64>
    bmodelica.raw_return %0 : tensor<3x5xf64>
}

// -----

bmodelica.raw_function @dynamicTensorResult(%arg0: index) -> tensor<3x?xf64> {
    %0 = tensor.empty(%arg0) : tensor<3x?xf64>
    bmodelica.raw_return %0 : tensor<3x?xf64>
}
