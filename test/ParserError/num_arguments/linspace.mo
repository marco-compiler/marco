// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function linspace at line 11, column 10. Expected 3 argument(s) but got 1.

function foo
    input Integer start;
    input Integer stop;
    input Integer n;
    output Real[:] y;
algorithm
    y := linspace(start);
end foo;
