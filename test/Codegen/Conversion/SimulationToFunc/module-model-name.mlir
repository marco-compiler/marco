// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// CHECK:   llvm.mlir.global internal constant @modelName("Test\00")

// CHECK:       func.func @getModelName() -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[addr:.*]] = llvm.mlir.addressof @modelName : !llvm.ptr<array<5 x i8>>
// CHECK-DAG:       %[[zero:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      %[[ptr:.*]] = llvm.getelementptr %[[addr]][%[[zero]], %[[zero]]] : (!llvm.ptr<array<5 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-NEXT:      return %[[ptr]] : !llvm.ptr<i8>
// CHECK-NEXT:  }

simulation.module attributes {modelName = "Test"} {

}
