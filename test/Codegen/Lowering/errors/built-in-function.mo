// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'sinx' was not declared in this scope
// CHECK-SAME: ; did you mean 'sin'?

function Foo
    input Real x;
    output Real y;
algorithm
    y := sinx(x);
end Foo;
