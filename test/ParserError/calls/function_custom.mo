// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown function identifier foO.

function foo
    input Integer[:] x;
    output Integer[:,:] y;
algorithm
    y := diagonal(x);
end foo;

function bar
    input Integer[:] x;
    output Integer[:,:] y;
algorithm
    y := foO(x);
end bar;

