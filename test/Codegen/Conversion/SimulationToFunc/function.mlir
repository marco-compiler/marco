// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// CHECK:       func.func @foo() {
// CHECK:           return
// CHECK-NEXT:  }

simulation.function @foo() {
    simulation.return
}

// -----

// CHECK:       func.func @foo(%[[arg:.*]]: f64) -> f64 {
// CHECK:           return %[[arg]]
// CHECK-NEXT:  }

simulation.function @foo(%arg0: f64) -> f64 {
    simulation.return %arg0 : f64
}
