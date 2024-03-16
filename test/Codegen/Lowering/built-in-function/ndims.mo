// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.ndims
// CHECK-SAME: <?x?x!modelica.int> -> !modelica.int

function foo
    input Integer[:,:] x;
    output Integer y;
algorithm
    y := ndims(x);
end foo;
