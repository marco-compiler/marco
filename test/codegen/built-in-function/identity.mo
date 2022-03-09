// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %arg0 : !modelica.int
// CHECK: modelica.identity %arg0 : !modelica.int -> !modelica.array<heap, ?x?x!modelica.int>

function foo
    input Integer x;
    output Integer[:,:] y;

algorithm
    y := identity(x);
end foo;
