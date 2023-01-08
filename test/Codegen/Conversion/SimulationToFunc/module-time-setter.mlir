// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// CHECK:       func.func @setTime(%[[opaquePtr:.*]]: !llvm.ptr<i8>, %[[time:.*]]: f64) {
// CHECK-NEXT:      %[[ptr1:.*]] = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK-NEXT:      %[[data:.*]] = llvm.load %[[ptr1]] : !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK-NEXT:      %[[newData:.*]] = llvm.insertvalue %[[time]], %[[data]][1] : !llvm.struct<(ptr<i8>, f64)>
// CHECK-NEXT:      %[[ptr2:.*]] = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK-NEXT:      llvm.store %[[newData]], %[[ptr2]] : !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK-NEXT:      return
// CHECK-NEXT:  }

simulation.module {

}
