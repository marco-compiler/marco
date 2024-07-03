// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function foo.

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
