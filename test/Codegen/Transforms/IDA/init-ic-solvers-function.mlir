// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_ic
// CHECK:       simulation.function @initICSolvers() {
// CHECK:           ida.create @ida_ic
// CHECK-DAG:       ida.set_start_time @ida_ic {time = 0.000000e+00 : f64}
// CHECK-DAG:       ida.set_end_time @ida_ic {time = 0.000000e+00 : f64}
// CHECK:           simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
