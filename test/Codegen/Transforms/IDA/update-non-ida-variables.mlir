// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK:       runtime.function @updateNonIDAVariables() {
// CHECK-NEXT:      runtime.return
// CHECK-NEXT:  }

bmodelica.model @Test {

}
