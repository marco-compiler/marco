// RUN: modelica-opt %s --split-input-file --euler-forward | FileCheck %s

// Empty model.

// CHECK:       simulation.function @calcIC() {
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
