// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_main
// CHECK:       runtime.dynamic_model_end {
// CHECK-NEXT:      ida.free @ida_main
// CHECK-NEXT:  }

modelica.model @Test {

}
