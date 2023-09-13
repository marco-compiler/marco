// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// CHECK:   llvm.mlir.global internal constant @modelName("Test\00")

// CHECK:       func.func @getModelName() -> !llvm.ptr {
// CHECK-DAG:       %[[addr:.*]] = llvm.mlir.addressof @modelName : !llvm.ptr<array<5 x i8>>
// CHECK-DAG:       %[[zero:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[ptr:.*]] = llvm.getelementptr %[[addr]][%[[zero]], %[[zero]]] : (!llvm.ptr<array<5 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           %[[opaque_ptr:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<i8> to !llvm.ptr
// CHECK:           return %[[opaque_ptr]] : !llvm.ptr
// CHECK-NEXT:  }

simulation.model_name "Test"
