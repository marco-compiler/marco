// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.diagonal
// CHECK-SAME: !modelica.array<?x!modelica.int> -> !modelica.array<?x?x!modelica.int>

function foo
    input Integer[:] x;
    output Integer[:,:] y;
algorithm
    y := diagonal(x);
end foo;
