// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_main
// CHECK:       simulation.function @deinitMainSolvers() {
// CHECK-NEXT:      ida.free @ida_main
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
