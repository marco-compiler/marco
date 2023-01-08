// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// CHECK:       func.func @getTime(%[[opaquePtr:.*]]: !llvm.ptr<i8>) -> f64 {
// CHECK-NEXT:      %[[ptr:.*]] = llvm.bitcast %[[opaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK-NEXT:      %[[data:.*]] = llvm.load %[[ptr]] : !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK-NEXT:      %[[time:.*]] = llvm.extractvalue %[[data]][1] : !llvm.struct<(ptr<i8>, f64)>
// CHECK-NEXT:      return %[[time]]
// CHECK-NEXT:  }

simulation.module {

}
