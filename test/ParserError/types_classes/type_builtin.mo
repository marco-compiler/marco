// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown type or class identifier Rel.

function Foo
    input Rel x;
    output Real y;
algorithm
    y := x;
end Foo;
