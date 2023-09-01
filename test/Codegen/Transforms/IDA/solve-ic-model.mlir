// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_ic
// CHECK:       simulation.function @solveICModel() {
// CHECK:           ida.calc_ic @ida_ic
// CHECK:           simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
