// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Unknown variable identifier fooo.

model M
  record R
    Real x;
    Real y;
  end R;

  Real foo;

  R[3] r(x(start = {fooo, 2.0, 3.0}), y(start = {4.0, 5.0, 6.0}));
end M;
