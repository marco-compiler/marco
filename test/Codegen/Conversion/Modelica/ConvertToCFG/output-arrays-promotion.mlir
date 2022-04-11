// RUN: modelica-opt %s --split-input-file --convert-modelica-to-cfg | FileCheck %s

// Static output arrays can be moved (if having an allowed size)

// CHECK-LABEL: @callee
// CHECK-SAME: (%[[X:[a-zA-Z0-9]*]]: !modelica.array<3x!modelica.int>) -> !modelica.array<?x!modelica.int>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.alloc
// CHECK-SAME: !modelica.array<?x!modelica.int>
// CHECK: return %[[Y]] : !modelica.array<?x!modelica.int>

modelica.function @callee : () -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>) {
    %0 = modelica.member_create @x : !modelica.member<3x!modelica.int, output>
    %1 = constant 2 : index
    %2 = modelica.member_create @y %1 : !modelica.member<?x!modelica.int, output>
}

// CHECK-LABEL: @caller
// CHECK: %[[X:[a-zA-Z0-9]*]] = modelica.alloc
// CHECK-SAME: !modelica.array<3x!modelica.int>
// CHECK: call @callee(%[[X]])
// CHECK-SAME: (!modelica.array<3x!modelica.int>) -> !modelica.array<?x!modelica.int>

modelica.function @caller : () -> () {
    %0:2 = modelica.call @callee() : () -> (!modelica.array<3x!modelica.int>, !modelica.array<?x!modelica.int>)
}
