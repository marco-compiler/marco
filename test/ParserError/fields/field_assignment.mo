// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Unknown field identifier fild.

record RECORD
    Real field;
end RECORD;

function Foo
    input RECORD x;
    output Real y;
algorithm
    y := x.fild;
end Foo;