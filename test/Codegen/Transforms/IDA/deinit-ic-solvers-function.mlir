// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_ic
// CHECK:       simulation.function @deinitICSolvers() {
// CHECK-NEXT:      ida.free @ida_ic
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
