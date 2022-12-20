// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false})" | FileCheck %s

// Scalar variable

// CHECK:       modelica.raw_function @foo(%{{.*}}: !modelica.int) {
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : (!modelica.int) -> () {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, input>
}

// -----

// Static array

// CHECK:       modelica.raw_function @foo(%{{.*}}: !modelica.array<3x2x!modelica.int>) {
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : (!modelica.array<3x2x!modelica.int>) -> () {
    %0 = modelica.member_create @x : !modelica.member<3x2x!modelica.int, input>
}

// -----

// Dynamic array

// CHECK:       modelica.raw_function @foo(%{{.*}}: !modelica.array<3x?x!modelica.int>) {
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @foo : (!modelica.array<3x?x!modelica.int>) -> () {
    %0 = modelica.member_create @x : !modelica.member<3x?x!modelica.int, input>
}
