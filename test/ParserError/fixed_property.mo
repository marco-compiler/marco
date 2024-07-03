// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Invalid fixed property for variable x.

model Test
    Boolean e;
    Real x(start = 5, fixed = e);
end Test;
