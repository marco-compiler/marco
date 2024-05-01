// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: %[[range:.*]] = bmodelica.range %{{.*}}, %{{.*}}, %{{.*}}
// CHECK-DAG: %[[begin:.*]] = bmodelica.range_begin %[[range]]
// CHECK-DAG: %[[begin_casted:.*]] = builtin.unrealized_conversion_cast %[[begin]] : !bmodelica.int to i64
// CHECK-DAG: %[[end:.*]] = bmodelica.range_end %[[range]]
// CHECK-DAG: %[[end_casted:.*]] = builtin.unrealized_conversion_cast %[[end]] : !bmodelica.int to i64
// CHECK-DAG: %[[step:.*]] = bmodelica.range_step %[[range]]
// CHECK-DAG: %[[step_casted:.*]] = builtin.unrealized_conversion_cast %[[step]] : !bmodelica.int to i64
// CHECK: %[[diff:.*]] = arith.subi %[[end_casted]], %[[begin_casted]]
// CHECK: %[[div:.*]] = arith.divsi %[[diff]], %[[step_casted]]
// CHECK: %[[div_casted:.*]] = arith.index_cast %[[div]]
// CHECK: %[[one:.*]] = arith.constant 1
// CHECK: %[[add:.*]] = arith.addi %[[div_casted]], %[[one]]
// CHECK: return %[[add]]

func.func @foo(%arg0: !bmodelica.int, %arg1: !bmodelica.int, %arg2: !bmodelica.int) -> index {
    %0 = bmodelica.range %arg0, %arg1, %arg2 : (!bmodelica.int, !bmodelica.int, !bmodelica.int) -> !bmodelica<range !bmodelica.int>
    %1 = bmodelica.range_size %0 : !bmodelica<range !bmodelica.int>
    func.return %1 : index
}
