// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'x' was not declared in this scope
// CHECK-SAME: ; did you mean 'y'?

function Foo
    output Real y;
algorithm
    y := x;
end Foo;
