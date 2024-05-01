// RUN: modelica-opt %s --split-input-file --euler-forward | FileCheck %s

// Empty model.

// CHECK:       runtime.function @updateNonStateVariables() {
// CHECK:           runtime.return
// CHECK-NEXT:  }

bmodelica.model @Test {

}
