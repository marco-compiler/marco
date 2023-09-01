// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_main
// CHECK:       simulation.function @getIDATime() -> f64 {
// CHECK-NEXT:      %[[result:.*]] = ida.get_current_time @ida_main : f64
// CHECK-NEXT:      simulation.return %[[result]]
// CHECK-NEXT:  }

modelica.model @Test {

}
