// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: 'x' was not declared in this scope

model M
equation
    x = 0;
end M;
