// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'Bool' was not declared in this scope
// CHECK-SAME: ; did you mean 'Boolean'?

function Foo
    output Bool y;
algorithm
    y := 0;
end Foo;
