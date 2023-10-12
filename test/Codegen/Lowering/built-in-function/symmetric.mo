// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.symmetric
// CHECK-SAME: !modelica.array<?x?x!modelica.int> -> !modelica.array<?x?x!modelica.int>

function foo
    input Integer[:,:] x;
    output Integer[:,:] y;
algorithm
    y := symmetric(x);
end foo;