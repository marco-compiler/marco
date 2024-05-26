// RUN: modelica-opt %s --split-input-file --heap-functions-replacement | FileCheck %s

// CHECK-DAG: llvm.func @marco_realloc(!llvm.ptr, i64) -> !llvm.ptr
// CHECK-DAG: llvm.call @marco_realloc

module {
    llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr

    func.func @foo(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.ptr {
        %0 = llvm.call @realloc(%arg0, %arg1) : (!llvm.ptr, i64) -> !llvm.ptr
        func.return %0 : !llvm.ptr
    }
}
