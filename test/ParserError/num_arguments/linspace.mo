// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: linspace: expected 3 argument(s) but got 1.

function foo
    input Integer start;
    input Integer stop;
    input Integer n;
    output Real[:] y;
algorithm
    y := linspace(start);
end foo;
