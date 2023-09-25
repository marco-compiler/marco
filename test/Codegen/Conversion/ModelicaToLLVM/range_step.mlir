// RUN: modelica-opt %s --split-input-file --convert-modelica-to-llvm | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: %[[range:.*]] = modelica.range %{{.*}}, %{{.*}}, %{{.*}}
// CHECK: %[[range_casted:.*]] = builtin.unrealized_conversion_cast %[[range]] : !modelica<range !modelica.int> to !llvm.struct<(i64, i64, i64)>
// CHECK: %[[result:.*]] = llvm.extractvalue %[[range_casted]][2]
// CHECK: %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK: return %[[result_casted]]

func.func @foo(%arg0: !modelica.int, %arg1: !modelica.int, %arg2: !modelica.int) -> !modelica.int {
    %0 = modelica.range %arg0, %arg1, %arg2 : (!modelica.int, !modelica.int, !modelica.int) -> !modelica<range !modelica.int>
    %1 = modelica.range_step %0 : !modelica<range !modelica.int> -> !modelica.int
    func.return %1 : !modelica.int
}
