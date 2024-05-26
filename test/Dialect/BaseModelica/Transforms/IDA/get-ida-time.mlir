// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_main
// CHECK:       runtime.function @getIDATime() -> f64 {
// CHECK-NEXT:      %[[result:.*]] = ida.get_current_time @ida_main : f64
// CHECK-NEXT:      runtime.return %[[result]]
// CHECK-NEXT:  }

bmodelica.model @Test {

}
