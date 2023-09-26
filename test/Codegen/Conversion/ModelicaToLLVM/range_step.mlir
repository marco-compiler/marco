// RUN: modelica-opt %s --split-input-file --convert-modelica-to-llvm | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[range:.*]]: !modelica<range !modelica.int>)
// CHECK: %[[range_casted:.*]] = builtin.unrealized_conversion_cast %[[range]] : !modelica<range !modelica.int> to !llvm.struct<(i64, i64, i64)>
// CHECK: %[[result:.*]] = llvm.extractvalue %[[range_casted]][2]
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK: return %[[result_casted]]

func.func @foo(%arg0: !modelica<range !modelica.int>) -> !modelica.int {
    %0 = modelica.range_step %arg0 : !modelica<range !modelica.int> -> !modelica.int
    func.return %0 : !modelica.int
}
