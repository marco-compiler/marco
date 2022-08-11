// RUN: modelica-opt %s --split-input-file --convert-modelica-to-llvm | FileCheck %s

// CHECK: llvm.func @[[foo:.*]](i1) -> i1
// CHECK-LABEL: @i1
// CHECK-SAME: (%[[arg0:.*]]: i1) -> i1
// CHECK: %[[result:.*]] = llvm.call @[[foo]](%[[arg0]]) : (i1) -> i1
// CHECK: return %[[result]]

modelica.runtime_function @foo : (i1) -> i1

func.func @i1(%arg0: i1) -> (i1) {
    %0 = modelica.call @foo(%arg0) : (i1) -> i1
    func.return %0 : i1
}

// -----

// CHECK: llvm.func @[[foo:.*]](i32) -> i32
// CHECK-LABEL: @i32
// CHECK-SAME: (%[[arg0:.*]]: i32) -> i32
// CHECK: %[[result:.*]] = llvm.call @[[foo]](%[[arg0]]) : (i32) -> i32
// CHECK: return %[[result]]

modelica.runtime_function @foo : (i32) -> i32

func.func @i32(%arg0: i32) -> (i32) {
    %0 = modelica.call @foo(%arg0) : (i32) -> i32
    func.return %0 : i32
}

// -----

// CHECK: llvm.func @[[foo:.*]](i64) -> i64
// CHECK-LABEL: @i64
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i64
// CHECK: %[[result:.*]] = llvm.call @[[foo]](%[[arg0]]) : (i64) -> i64
// CHECK: return %[[result]]

modelica.runtime_function @foo : (i64) -> i64

func.func @i64(%arg0: i64) -> (i64) {
    %0 = modelica.call @foo(%arg0) : (i64) -> i64
    func.return %0 : i64
}

// -----

// CHECK: llvm.func @[[foo:.*]](f32) -> f32
// CHECK-LABEL: @f32
// CHECK-SAME: (%[[arg0:.*]]: f32) -> f32
// CHECK: %[[result:.*]] = llvm.call @[[foo]](%[[arg0]]) : (f32) -> f32
// CHECK: return %[[result]]

modelica.runtime_function @foo : (f32) -> f32

func.func @f32(%arg0: f32) -> (f32) {
    %0 = modelica.call @foo(%arg0) : (f32) -> f32
    func.return %0 : f32
}

// -----

// CHECK: llvm.func @[[foo:.*]](f64) -> f64
// CHECK-LABEL: @f64
// CHECK-SAME: (%[[arg0:.*]]: f64) -> f64
// CHECK: %[[result:.*]] = llvm.call @[[foo]](%[[arg0]]) : (f64) -> f64
// CHECK: return %[[result]]

modelica.runtime_function @foo : (f64) -> f64

func.func @f64(%arg0: f64) -> (f64) {
    %0 = modelica.call @foo(%arg0) : (f64) -> f64
    func.return %0 : f64
}

// -----

// CHECK: llvm.func @[[foo:.*]](!llvm.ptr<struct<(i64, ptr<i8>)>>)
// CHECK-LABEL: @memref
// CHECK-SAME: (%[[arg0:.*]]: memref<*xi64>)
// CHECK: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : memref<*xi64> to !llvm.struct<(i64, ptr<i8>
// CHECK: %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[ptr:.*]] = llvm.alloca %[[one]] x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
// CHECK: llvm.call @[[foo]](%[[ptr]]) : (!llvm.ptr<struct<(i64, ptr<i8>)>>) -> ()
// CHECK: return

modelica.runtime_function @foo : (memref<*xi64>) -> ()

func.func @memref(%arg0: memref<*xi64>) -> () {
    modelica.call @foo(%arg0) : (memref<*xi64>) -> ()
    func.return
}
