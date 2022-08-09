// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK:       func.func @foo(%[[arg0:.*]]: i1) -> i1 {
// CHECK-NEXT:      return %[[arg0]] : i1
// CHECK-NEXT:  }

modelica.raw_function @foo(%arg0: !modelica.bool) -> !modelica.bool {
    modelica.raw_return %arg0 : !modelica.bool
}

// -----

// CHECK:       func.func @foo(%[[arg0:.*]]: i64) -> i64 {
// CHECK-NEXT:      return %[[arg0]] : i64
// CHECK-NEXT:  }

modelica.raw_function @foo(%arg0: !modelica.int) -> !modelica.int {
    modelica.raw_return %arg0 : !modelica.int
}

// -----

// CHECK:       func.func @foo(%[[arg0:.*]]: f64) -> f64 {
// CHECK-NEXT:      return %[[arg0]] : f64
// CHECK-NEXT:  }

modelica.raw_function @foo(%arg0: !modelica.real) -> !modelica.real {
    modelica.raw_return %arg0 : !modelica.real
}

// -----

// CHECK:       func.func @foo(%[[arg0:.*]]: memref<5x3xi64>) -> memref<5x3xi64> {
// CHECK-NEXT:      return %[[arg0]] : memref<5x3xi64>
// CHECK-NEXT:  }

modelica.raw_function @foo(%arg0: !modelica.array<5x3x!modelica.int>) -> !modelica.array<5x3x!modelica.int> {
    modelica.raw_return %arg0 : !modelica.array<5x3x!modelica.int>
}
