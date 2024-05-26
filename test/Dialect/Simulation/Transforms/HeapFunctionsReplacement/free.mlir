// RUN: modelica-opt %s --split-input-file --heap-functions-replacement | FileCheck %s

// CHECK-DAG: llvm.func @marco_free(!llvm.ptr)
// CHECK-DAG: llvm.call @marco_free

module {
    llvm.func @free(!llvm.ptr)

    func.func @foo(%arg0: !llvm.ptr) {
        llvm.call @free(%arg0) : (!llvm.ptr) -> ()
        func.return
    }
}
