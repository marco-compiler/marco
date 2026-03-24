// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'Bar' was not declared in this scope
// CHECK-SAME: did you mean 'Foo'?

function Foo
    output Integer x;
algorithm
    x := Bar();
end Foo;
