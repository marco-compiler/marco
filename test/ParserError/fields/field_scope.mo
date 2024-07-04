// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Unknown field identifier fild.

record RECORD1
    Real field;
end RECORD1;

record RECORD2
    Real fild;
end RECORD2;

function Foo
    input RECORD1 x;
    output Real y;
algorithm
    y := x.fild;
end Foo;