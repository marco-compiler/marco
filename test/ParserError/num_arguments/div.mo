// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function div at line 9, column 10. Expected 2 argument(s) but got 3.
function foo
    input Integer x;
    input Integer y;
    output Integer z;
algorithm
    z := div(x, y, x);
end foo;
