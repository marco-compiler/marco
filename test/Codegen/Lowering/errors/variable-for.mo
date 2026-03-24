// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'j' was not declared in this scope
// CHECK-SAME: ; did you mean 'i'?

function Foo
    output Real[10] result;
algorithm
    for i in 1:10 loop
        result[j] := 0;
    end for;
end Foo;
