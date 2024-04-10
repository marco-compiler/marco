// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown variable identifier fooo at line 13, column 21. Did you mean foo?

model M
  record R
    Real x;
    Real y;
  end R;

  Real foo;

  R[3] r(x(start = {fooo, 2.0, 3.0}), y(start = {4.0, 5.0, 6.0}));
end M;
