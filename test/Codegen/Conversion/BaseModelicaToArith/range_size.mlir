// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: %[[range:.*]] = bmodelica.range %{{.*}}, %{{.*}}, %{{.*}}
// CHECK-DAG: %[[begin:.*]] = bmodelica.range_begin %[[range]]
// CHECK-DAG: %[[end:.*]] = bmodelica.range_end %[[range]]
// CHECK-DAG: %[[step:.*]] = bmodelica.range_step %[[range]]
// CHECK: %[[diff:.*]] = arith.subi %[[end]], %[[begin]]
// CHECK: %[[div:.*]] = arith.divsi %[[diff]], %[[step]]
// CHECK: %[[div_casted:.*]] = arith.index_cast %[[div]]
// CHECK: %[[one:.*]] = arith.constant 1
// CHECK: %[[add:.*]] = arith.addi %[[div_casted]], %[[one]]
// CHECK: return %[[add]]

func.func @foo(%arg0: i64, %arg1: i64, %arg2: i64) -> index {
    %0 = bmodelica.range %arg0, %arg1, %arg2 : (i64, i64, i64) -> !bmodelica<range i64>
    %1 = bmodelica.range_size %0 : !bmodelica<range i64>
    func.return %1 : index
}
