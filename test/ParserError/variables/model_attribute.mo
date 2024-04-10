// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown variable identifier att at line 7, column 20. Did you mean attr?

model Test
    Real attr;
    Real x(start = att);
end Test;
