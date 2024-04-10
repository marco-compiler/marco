// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown field identifier fild at line 13, column 12. Did you mean field?

record RECORD
    Real field;
end RECORD;

function Foo
    input RECORD x;
    output Real y;
algorithm
    y := x.fild;
end Foo;