// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function foo at line 16, column 10. Expected 1 argument(s) but got 2.

function foo
    input Real x;
    output Real y;
algorithm
    y := x;
end foo;

function bar
    input Real x;
    output Real y;
algorithm
    y := foo(x, x);
end bar;
