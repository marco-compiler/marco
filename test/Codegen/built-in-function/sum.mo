// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.sum
// CHECK-SAME: !modelica.array<?x!modelica.int> -> !modelica.int

function foo
    input Integer[:] x;
    output Integer y;

algorithm
    y := sum(x);
end foo;