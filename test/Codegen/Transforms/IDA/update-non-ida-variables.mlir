// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK:       simulation.function @updateNonIDAVariables() {
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
