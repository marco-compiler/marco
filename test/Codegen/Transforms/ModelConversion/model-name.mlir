// RUN: modelica-opt %s --split-input-file --pass-pipeline="convert-model{model=Test}" | FileCheck %s

// Model name getter

// CHECK:   llvm.mlir.global internal constant @modelName("Test")

// CHECK:   func.func @getModelName() -> !llvm.ptr<i8> {
// CHECK:       %[[addr:.*]] = llvm.mlir.addressof @modelName : !llvm.ptr<array<4 x i8>>
// CHECK:       %[[zero:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:       %[[ptr:.*]] = llvm.getelementptr %[[addr]][%[[zero]], %[[zero]]] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:       return %[[ptr]] : !llvm.ptr<i8>
// CHECK:   }

modelica.model @Test {

} body {

}

