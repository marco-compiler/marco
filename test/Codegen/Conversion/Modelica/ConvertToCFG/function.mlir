// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cfg | FileCheck %s

// CHECK:      func @foo(%arg0: !modelica.int) -> !modelica.int {
// CHECK-NEXT:      %0 = modelica.alloca  : !modelica.array<!modelica.int>
// CHECK-NEXT:      br ^[[out:[a-zA-Z0-9]*]]
// CHECK-NEXT: ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      %1 = modelica.load %0[] : !modelica.array<!modelica.int>
// CHECK-NEXT:      return %1 : !modelica.int
// CHECK-NEXT: }

modelica.function @foo : (!modelica.int) -> (!modelica.int) {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int, output>
}
