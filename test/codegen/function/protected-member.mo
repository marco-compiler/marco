// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %arg0 : !modelica.array<?x!modelica.int>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<stack, !modelica.int>
// CHECK: %[[Z:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "z"
// CHECK-SAME: !modelica.member<stack, 2x!modelica.int>

function foo
    input Integer[:] x;
    output Integer y;

protected
    Integer[2] z;

algorithm

end foo;
