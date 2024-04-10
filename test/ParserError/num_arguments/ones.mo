// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function ones at line 10, column 10. Expected at least 1 argument(s) but got 0.

function foo
    input Integer n1;
    input Integer n2;
    output Real[:] y;
algorithm
    y := ones();
end foo;
