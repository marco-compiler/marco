// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cf | FileCheck %s

// CHECK:       modelica.raw_function @foo(%{{.*}}: !modelica.int) -> !modelica.int {
// CHECK:           %[[y:.*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      %[[result:.*]] = modelica.load %[[y]][] : !modelica.array<!modelica.int>
// CHECK-NEXT:      modelica.raw_return %[[result]]
// CHECK-NEXT:  }

modelica.function @foo : (!modelica.int) -> (!modelica.int) {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, input>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int, output>
}
