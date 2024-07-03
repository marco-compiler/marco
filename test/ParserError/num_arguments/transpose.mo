// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function transpose.

function foo
    input Integer[:,:] x;
    output Integer[:,:] y;
algorithm
    y := transpose(x, x);
end foo;