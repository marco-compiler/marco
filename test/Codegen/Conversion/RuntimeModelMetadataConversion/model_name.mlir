// RUN: modelica-opt %s --split-input-file --convert-runtime-model-metadata | FileCheck %s

// CHECK:   llvm.mlir.global internal constant @modelName("Test\00")

// CHECK:       func.func @getModelName() -> !llvm.ptr {
// CHECK-DAG:       %[[addr:.*]] = llvm.mlir.addressof @modelName : !llvm.ptr
// CHECK:           %[[ptr:.*]] = llvm.getelementptr %[[addr]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<5 x i8>
// CHECK:           return %[[ptr]] : !llvm.ptr
// CHECK-NEXT:  }

runtime.model_name "Test"
