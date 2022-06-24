// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// CHECK:      func @foo(%arg0: i64) -> i64 {
// CHECK:      %[[Y:[a-zA-Z0-9]*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK-NEXT:      cf.br ^[[out:[a-zA-Z0-9]*]]
// CHECK-NEXT: ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      %[[Y_LOAD:[a-zA-Z0-9]*]] = modelica.load %[[Y]][] : !modelica.array<!modelica.int>
// CHECK-NEXT:      %[[Y_CAST:[a-zA-Z0-9]*]] = builtin.unrealized_conversion_cast %[[Y_LOAD]] : !modelica.int to i64
// CHECK-NEXT:      return %[[Y_CAST]]
// CHECK-NEXT: }

modelica.function @foo : (!modelica.int) -> (!modelica.int) {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int, output>
}
