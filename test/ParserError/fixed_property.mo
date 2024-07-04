// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Invalid fixed property for variable x.

model Test
    Boolean e;
    Real x(start = 5, fixed = e);
end Test;
