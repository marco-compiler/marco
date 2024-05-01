// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-llvm | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int, %[[arg2:.*]]: !bmodelica.int)
// CHECK-DAG: %[[lowerBound:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[upperBound:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK-DAG: %[[step:.*]] = builtin.unrealized_conversion_cast %[[arg2]] : !bmodelica.int to i64
// CHECK: %[[undef:.*]] = llvm.mlir.undef : !llvm.struct<(i64, i64, i64)>
// CHECK: %[[insert_0:.*]] = llvm.insertvalue %[[lowerBound]], %[[undef]][0]
// CHECK: %[[insert_1:.*]] = llvm.insertvalue %[[upperBound]], %[[insert_0]][1]
// CHECK: %[[insert_2:.*]] = llvm.insertvalue %[[step]], %[[insert_1]][2]
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[insert_2]] : !llvm.struct<(i64, i64, i64)> to !bmodelica<range !bmodelica.int>
// CHECK: return %[[result]]

func.func @foo(%arg0: !bmodelica.int, %arg1: !bmodelica.int, %arg2: !bmodelica.int) -> !bmodelica<range !bmodelica.int> {
    %0 = bmodelica.range %arg0, %arg1, %arg2 : (!bmodelica.int, !bmodelica.int, !bmodelica.int) -> !bmodelica<range !bmodelica.int>
    func.return %0 : !bmodelica<range !bmodelica.int>
}
