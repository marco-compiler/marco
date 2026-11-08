// RUN: not marco -mc1 -emit-mlir-modelica --model=M -o - %s 2>&1 | FileCheck %s

// The Modelica dialect is available only with the native frontend.

// CHECK: The Modelica dialect can be emitted only by the native frontend

model M
end M;
