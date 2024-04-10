// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown field identifier fild at line 17, column 12. Did you mean field?

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