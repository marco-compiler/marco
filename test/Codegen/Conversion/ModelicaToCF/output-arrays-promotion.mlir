// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// Static output arrays can be moved (if having an allowed size)

// CHECK-LABEL: @callee
// CHECK-SAME: (%[[x:.*]]: !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>) -> !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: %[[y:.*]] = modelica.alloc
// CHECK-SAME: !modelica.array<?x!modelica.int>
// CHECK: %[[y_cast:.*]] = builtin.unrealized_conversion_cast %[[y]] : !modelica.array<?x!modelica.int> to !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: modelica.raw_return %[[y_cast]]

modelica.function @callee : () -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>) {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int, output>
    %1 = arith.constant 2 : index
    %2 = modelica.member_create @y %1 : !modelica.member<?x!modelica.int, output>
}

// CHECK-LABEL: @caller
// CHECK: %[[x:.*]] = modelica.alloc
// CHECK-SAME: !modelica.array<3x!modelica.int>
// CHECK: %[[x_cast:.*]] = builtin.unrealized_conversion_cast %[[x]] : !modelica.array<3x!modelica.int> to !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: modelica.call @callee(%[[x_cast]])
// CHECK-SAME: (!llvm.struct<(ptr<i64>, i64, array<1 x i64>)>) -> !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>

modelica.function @caller : () -> () {
    %0:2 = modelica.call @callee() : () -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>)
}
