// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-SAME: %arg0 : !modelica.array<?x!modelica.int>
// CHECK: modelica.diagonal %arg0 : !modelica.array<?x!modelica.int> -> !modelica.array<heap, ?x?x!modelica.int>

function foo
    input Integer[:] x;
    output Integer[:,:] y;

algorithm
    y := diagonal(x);
end foo;
