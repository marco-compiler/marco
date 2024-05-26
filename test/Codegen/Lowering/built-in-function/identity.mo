// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.identity
// CHECK-SAME: !bmodelica.int -> tensor<?x?x!bmodelica.int>

function foo
    input Integer x;
    output Integer[:,:] y;
algorithm
    y := identity(x);
end foo;
