// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo

// CHECK: modelica.member_create
// CHECK-SAME: name = "x"
// CHECK-SAME: !modelica.member<?x!modelica.int, input>

// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<!modelica.int, output>

// CHECK: modelica.member_create
// CHECK-SAME: name = "z"
// CHECK-SAME: !modelica.member<2x!modelica.int>

function foo
    input Integer[:] x;
    output Integer y;

protected
    Integer[2] z;

algorithm

end foo;
