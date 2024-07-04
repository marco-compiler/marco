// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Unknown type or class identifier RECRD.

record RECORD
    Real x;
end RECORD;

function Foo
    input RECRD r;
    output Real x;
algorithm
    x := r.x;
end Foo;
