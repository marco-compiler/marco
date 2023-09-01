// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_main
// CHECK:       simulation.function @updateIDAVariables() {
// CHECK:           ida.step @ida_main
// CHECK:           simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
