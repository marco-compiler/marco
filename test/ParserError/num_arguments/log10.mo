// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function log10 at line 9, column 10. Expected 1 argument(s) but got 2.

function foo
    input Real x;
    output Real y;
algorithm
    y := log10(x, x);
end foo;
