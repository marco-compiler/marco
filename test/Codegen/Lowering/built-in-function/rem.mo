// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.rem
// CHECK-SAME: (!modelica.int, !modelica.int) -> !modelica.int

function foo
    input Integer x;
    input Integer y;
    output Integer z;
algorithm
    z := rem(x, y);
end foo;
