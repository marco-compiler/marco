// RUN: modelica-opt %s --split-input-file --convert-modelica-to-llvm | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int, %[[arg2:.*]]: !modelica.int)
// CHECK-DAG: %[[lowerBound:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[upperBound:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK-DAG: %[[step:.*]] = builtin.unrealized_conversion_cast %[[arg2]] : !modelica.int to i64
// CHECK: %[[undef:.*]] = llvm.mlir.undef : !llvm.struct<(i64, i64, i64)>
// CHECK: %[[insert_0:.*]] = llvm.insertvalue %[[lowerBound]], %[[undef]][0]
// CHECK: %[[insert_1:.*]] = llvm.insertvalue %[[upperBound]], %[[insert_0]][1]
// CHECK: %[[insert_2:.*]] = llvm.insertvalue %[[step]], %[[insert_1]][2]
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[insert_2]] : !llvm.struct<(i64, i64, i64)> to !modelica<range !modelica.int>
// CHECK: return %[[result]]

func.func @foo(%arg0: !modelica.int, %arg1: !modelica.int, %arg2: !modelica.int) -> !modelica<range !modelica.int> {
    %0 = modelica.range %arg0, %arg1, %arg2 : (!modelica.int, !modelica.int, !modelica.int) -> !modelica<range !modelica.int>
    func.return %0 : !modelica<range !modelica.int>
}
