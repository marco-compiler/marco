// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'Rea' was not declared in this scope
// CHECK-SAME: ; did you mean 'Real'?

function Foo
    output Rea y;
algorithm
    y := 0;
end Foo;
