// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_main
// CHECK:       runtime.function @updateIDAVariables() {
// CHECK:           ida.step @ida_main
// CHECK:           runtime.return
// CHECK-NEXT:  }

modelica.model @Test {

}
