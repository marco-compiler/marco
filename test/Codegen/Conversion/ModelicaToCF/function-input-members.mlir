// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false})" | FileCheck %s

// Scalar variable

// CHECK:       modelica.raw_function @scalarVariable(%{{.*}}: !modelica.int) {
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @scalarVariable : (!modelica.int) -> () {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, input>
}

// -----

// Load of a scalar variable

// CHECK:   modelica.raw_function @scalarVariableLoad(%[[x:.*]]: !modelica.int) {
// CHECK:       %{{.*}} = modelica.neg %[[x]]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @scalarVariableLoad : (!modelica.int) -> () {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int, input>
    %1 = modelica.member_load %0 : !modelica.member<!modelica.int, input>
    %2 = modelica.neg %1 : !modelica.int -> !modelica.int
}

// -----

// Static array

// CHECK:       modelica.raw_function @staticArray(%{{.*}}: !modelica.array<3x2x!modelica.int>) {
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @staticArray : (!modelica.array<3x2x!modelica.int>) -> () {
    %0 = modelica.member_create @x : !modelica.member<3x2x!modelica.int, input>
}

// -----

// Load of a static array

// CHECK:   modelica.raw_function @staticArrayLoad(%[[x:.*]]: !modelica.array<3x2x!modelica.int>) {
// CHECK:       modelica.load %[[x]][{{.*}}, {{.*}}]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @staticArrayLoad : (!modelica.array<3x2x!modelica.int>) -> () {
    %0 = modelica.member_create @x : !modelica.member<3x2x!modelica.int, input>
    %1 = modelica.member_load %0 : !modelica.member<3x2x!modelica.int, input>
    %2 = arith.constant 0 : index
    %3 = modelica.load %1[%2, %2] : !modelica.array<3x2x!modelica.int>
}

// -----

// Dynamic array

// CHECK:       modelica.raw_function @dynamicArray(%{{.*}}: !modelica.array<3x?x!modelica.int>) {
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @dynamicArray : (!modelica.array<3x?x!modelica.int>) -> () {
    %0 = modelica.member_create @x : !modelica.member<3x?x!modelica.int, input>
}

// -----

// Load of a dynamic array

// CHECK:   modelica.raw_function @dynamicArrayLoad(%[[x:.*]]: !modelica.array<3x?x!modelica.int>) {
// CHECK:       modelica.load %[[x]][{{.*}}, {{.*}}]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @dynamicArrayLoad : (!modelica.array<3x?x!modelica.int>) -> () {
    %0 = modelica.member_create @x : !modelica.member<3x?x!modelica.int, input>
    %1 = modelica.member_load %0 : !modelica.member<3x?x!modelica.int, input>
    %2 = arith.constant 0 : index
    %3 = modelica.load %1[%2, %2] : !modelica.array<3x?x!modelica.int>
}
