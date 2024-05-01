// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-llvm | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[range:.*]]: !bmodelica<range !bmodelica.int>)
// CHECK: %[[range_casted:.*]] = builtin.unrealized_conversion_cast %[[range]] : !bmodelica<range !bmodelica.int> to !llvm.struct<(i64, i64, i64)>
// CHECK: %[[result:.*]] = llvm.extractvalue %[[range_casted]][2]
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @foo(%arg0: !bmodelica<range !bmodelica.int>) -> !bmodelica.int {
    %0 = bmodelica.range_step %arg0 : !bmodelica<range !bmodelica.int> -> !bmodelica.int
    func.return %0 : !bmodelica.int
}
