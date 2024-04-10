// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown type or class identifier Rel at line 6, column 11. Did you mean Real?

function Foo
    input Rel x;
    output Real y;
algorithm
    y := x;
end Foo;
