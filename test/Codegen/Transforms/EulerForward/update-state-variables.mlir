// RUN: modelica-opt %s --split-input-file --euler-forward | FileCheck %s

// Empty model.

// CHECK:       runtime.function @updateStateVariables(%[[timeStep:.*]]: f64) {
// CHECK:           runtime.return
// CHECK-NEXT:  }

modelica.model @Test {

}
