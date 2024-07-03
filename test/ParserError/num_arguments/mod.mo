// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function mod.

function foo
    input Integer x;
    input Integer y;
    output Integer z;
algorithm
    z := mod(x);
end foo;
