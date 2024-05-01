// RUN: modelica-opt %s --split-input-file --convert-runtime-to-func | FileCheck %s

// CHECK:       func.func @foo() {
// CHECK:           return
// CHECK-NEXT:  }

runtime.function @foo() {
    runtime.return
}

// -----

// CHECK:       func.func @foo(%[[arg:.*]]: f64) -> f64 {
// CHECK:           return %[[arg]]
// CHECK-NEXT:  }

runtime.function @foo(%arg0: f64) -> f64 {
    runtime.return %arg0 : f64
}
