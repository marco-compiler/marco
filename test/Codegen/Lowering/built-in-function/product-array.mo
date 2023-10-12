// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.product
// CHECK-SAME: !modelica.array<?x!modelica.int> -> !modelica.int

function foo
    input Integer[:] x;
    output Integer y;
algorithm
    y := product(x);
end foo;
