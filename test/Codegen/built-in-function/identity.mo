// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.identity
// CHECK-SAME: !modelica.int -> !modelica.array<?x?x!modelica.int>

function foo
    input Integer x;
    output Integer[:,:] y;

algorithm
    y := identity(x);
end foo;
