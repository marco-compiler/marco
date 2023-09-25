// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: %[[range:.*]] = modelica.range %{{.*}}, %{{.*}}, %{{.*}}
// CHECK-DAG: %[[begin:.*]] = modelica.range_begin %[[range]]
// CHECK-DAG: %[[begin_casted:.*]] = builtin.unrealized_conversion_cast %[[begin]] : !modelica.int to i64
// CHECK-DAG: %[[end:.*]] = modelica.range_end %[[range]]
// CHECK-DAG: %[[end_casted:.*]] = builtin.unrealized_conversion_cast %[[end]] : !modelica.int to i64
// CHECK-DAG: %[[step:.*]] = modelica.range_step %[[range]]
// CHECK-DAG: %[[step_casted:.*]] = builtin.unrealized_conversion_cast %[[step]] : !modelica.int to i64
// CHECK: %[[diff:.*]] = arith.subi %[[end_casted]], %[[begin_casted]]
// CHECK: %[[div:.*]] = arith.divsi %[[diff]], %[[step_casted]]
// CHECK: %[[div_casted:.*]] = arith.index_cast %[[div]]
// CHECK: %[[one:.*]] = arith.constant 1
// CHECK: %[[add:.*]] = arith.addi %[[div_casted]], %[[one]]
// CHECK: return %[[add]]

func.func @foo(%arg0: !modelica.int, %arg1: !modelica.int, %arg2: !modelica.int) -> index {
    %0 = modelica.range %arg0, %arg1, %arg2 : (!modelica.int, !modelica.int, !modelica.int) -> !modelica<range !modelica.int>
    %1 = modelica.range_size %0 : !modelica<range !modelica.int>
    func.return %1 : index
}
