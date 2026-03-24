// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'y' was not declared in this scope
// CHECK-SAME: did you mean 'x'?

record R1
    Real x;
end R1;

function Foo
    input R1 r1;
    output Real x;
algorithm
    x := r1.y;
end Foo;
