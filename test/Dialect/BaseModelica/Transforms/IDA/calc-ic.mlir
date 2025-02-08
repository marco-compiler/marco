// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// CHECK: ida.instance @ida_main
// CHECK:       runtime.function @calcIC() {
// CHECK:           ida.calc_ic @ida_main
// CHECK:           runtime.return
// CHECK-NEXT:  }

bmodelica.model @emptyModel {

}
