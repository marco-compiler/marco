// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'R2' was not declared in this scope
// CHECK-SAME: did you mean 'R1'?

record R1
end R1;

function Foo
    R2 r2;
end Foo;
