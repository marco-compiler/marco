// RUN: modelica-opt %s --split-input-file --heap-functions-replacement | FileCheck %s

// CHECK-DAG: llvm.func @marco_malloc(i64) -> !llvm.ptr
// CHECK-DAG: llvm.call @marco_malloc

module {
    llvm.func @malloc(i64) -> !llvm.ptr

    func.func @foo(%arg0: i64) -> !llvm.ptr {
        %0 = llvm.call @malloc(%arg0) : (i64) -> !llvm.ptr
        func.return %0 : !llvm.ptr
    }
}
