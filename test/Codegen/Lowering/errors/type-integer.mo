// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'Int' was not declared in this scope
// CHECK-SAME: ; did you mean 'Integer'?

function Foo
    output Int y;
algorithm
    y := 0;
end Foo;
