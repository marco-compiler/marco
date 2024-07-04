// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Unknown variable identifier att.

model Test
    Real attr;
    Real x(start = att);
end Test;
